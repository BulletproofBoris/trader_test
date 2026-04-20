# _tools/generate_features.py
# -*- coding: utf-8 -*-
"""
Объединенный генератор признаков с параллельной обработкой и прогресс-баром.
Запуск:
    python -m _tools.generate_features --universe_file universe.csv --workers 7
"""
from __future__ import annotations

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional
import inspect

import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from contextlib import contextmanager

# --- Подготовка путей для импортов пакета ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# --- Импорты из ядра проекта ---
from _core.paths import DATA_DIR, FEATURES_DIR_RAW, ensure_dirs, PROJECT_ROOT
from _core.data_loader import load_data
from _core.feature_generator import (
    create_cross_sectional_features,
    create_individual_features,
)

warnings.simplefilter("ignore", UserWarning)


# --- tqdm-joblib: прогресс-бар для Parallel ---
@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager для отображения прогресса joblib.Parallel с tqdm."""
    from joblib import parallel
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_cb = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


def _load_universe(universe_file: str) -> list[str]:
    f = (PROJECT_ROOT / universe_file) if not universe_file.startswith(str(PROJECT_ROOT)) else Path(universe_file)
    if not f.exists():
        raise FileNotFoundError(f"Не найден файл вселенной: {f}")
    df = pd.read_csv(f)
    if "ticker" in df.columns:
        vals = df["ticker"].dropna().astype(str).str.strip().tolist()
    else:
        vals = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    return [t for t in vals if t]


def _try_load_first(candidates: list[str]) -> Optional[pd.DataFrame]:
    for name in candidates:
        try:
            df = load_data(name)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            continue
    return None


def _build_external_data() -> pd.DataFrame:
    """
    Собираем внешний DataFrame с индексом datetime.
    ФИКС: для 'CBR_KEY_RATE' поддерживаем колонку 'key_rate' (переименовываем в 'close'),
    чтобы итоговое имя стало 'cbr_rate_close'.
    """
    sources: Dict[str, list[str]] = {
        "imoex":    ["IMOEX", "IMOEX.csv"],
        "usdrub":   ["USDRUB", "USDRUB.csv", "USD_RUB"],
        "brent":    ["BRENT", "BRENT.csv", "BR"],
        "sp500":    ["SP500", "SP500.csv", "^GSPC", "S&P500"],
        "vix":      ["VIX", "VIX.csv"],
        "cbr_rate": ["CBR_KEY_RATE", "CBR_KEY_RATE.csv", "KEY_RATE"],
    }
    frames = []
    for key, candidates in sources.items():
        df = _try_load_first(candidates)
        if df is None or df.empty:
            continue

        # Нормализуем колонку даты
        if "datetime" not in df.columns:
            for alt in ["date", "time", "Date", "DATE"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "datetime"})
                    break

        # Для ставки ЦБ: поддержка колонки 'key_rate' (заменим на 'close')
        if key == "cbr_rate":
            if "close" not in df.columns and "key_rate" in df.columns:
                df = df.rename(columns={"key_rate": "close"})

        # Общие требования
        if "datetime" not in df.columns or "close" not in df.columns:
            continue

        # Приведение типов и переименование цены -> <key>_close
        df = df[["datetime", "close"]].copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"]).sort_values("datetime")
        df = df.rename(columns={"close": f"{key}_close"}).set_index("datetime")
        frames.append(df)

    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="datetime"))

    ext = pd.concat(frames, axis=1, join="outer")
    ext = ext[~ext.index.duplicated(keep="first")]
    return ext


def _collect_all_data(tickers: list[str]) -> Dict[str, pd.DataFrame]:
    all_data: Dict[str, pd.DataFrame] = {}
    for t in tqdm(tickers, desc="Загрузка данных"):
        try:
            df = load_data(t)
            if df is not None and not df.empty:
                all_data[t] = df
        except Exception as e:
            warnings.warn(f"[{t}] не удалось загрузить: {e}")
    return all_data


def _call_create_individual_features(
    item: Tuple[str, pd.DataFrame],
    cs: Optional[pd.DataFrame],
    external_df: Optional[pd.DataFrame],
):
    """Безопасный вызов create_individual_features с учётом фактической сигнатуры в ядре."""
    sig = inspect.signature(create_individual_features)
    params = sig.parameters
    kwargs = {}
    if "external_data" in params:
        kwargs["external_data"] = external_df
    elif "external" in params:
        kwargs["external"] = external_df
    if "cs_features" in params:
        kwargs["cs_features"] = cs
    try:
        return create_individual_features(item, **kwargs)
    except TypeError:
        # Фоллбэк на позиционный вызов (если ядро не принимает kwargs)
        pos_args = [item]
        pos_param_names = [
            name for name, p in params.items()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ][1:]
        if pos_param_names:
            if len(pos_param_names) >= 1:
                pos_args.append(external_df)
            if len(pos_param_names) >= 2:
                pos_args.append(cs)
            return create_individual_features(*pos_args)
        raise


def _process_one(item: Tuple[str, pd.DataFrame], cs, external_df):
    """Параллельная обработка одного тикера."""
    try:
        t, df_feats = _call_create_individual_features(item, cs, external_df)
        return t, df_feats
    except Exception as e:
        warnings.warn(f"[{item[0]}] ошибка при расчёте признаков: {e}")
        return item[0], None


def main(args: argparse.Namespace) -> None:
    ensure_dirs()
    print("—" * 60)
    print("🚀 Генерация признаков (_tools.generate_features)")
    print(f"ROOT: {ROOT}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"FEATURES_DIR_RAW: {FEATURES_DIR_RAW}")
    print("—" * 60)

    tickers = _load_universe(args.universe_file)
    if not tickers:
        print("⚠️ Вселенная пуста — выходим.")
        return

    all_data = _collect_all_data(tickers)
    if not all_data:
        print("⚠️ Нет данных по тикерам — выходим.")
        return

    # 1) Кросс-секционные признаки
    cs = create_cross_sectional_features(all_data)

    # 2) Внешние ряды (в т.ч. key_rate -> cbr_rate_close)
    external_df = _build_external_data()

    # 3) Индивидуальные признаки — параллельно с прогресс-баром
    items = list(all_data.items())
    n_jobs = max(1, int(args.workers))
    print(f"Параллельная обработка: workers={n_jobs}")

    with tqdm_joblib(tqdm(desc="Создание индивидуальных признаков", total=len(items))):
        results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
            delayed(_process_one)(item, cs, external_df) for item in items
        )

    features: Dict[str, pd.DataFrame] = {t: df for t, df in results if df is not None and not df.empty}
    if not features:
        print("⚠️ Ни одного набора признаков не создано — выходим.")
        return

    # 4) Сохранение
    FEATURES_DIR_RAW.mkdir(parents=True, exist_ok=True)
    for t, df in tqdm(features.items(), desc="Сохранение файлов"):
        df.reset_index().to_csv(FEATURES_DIR_RAW / f"{t}.csv", index=False)

    print(f"✅ Готово. Сохранено файлов: {len(features)} → {FEATURES_DIR_RAW}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Объединенный генератор признаков (параллельный).")
    parser.add_argument("--universe_file", type=str, default="universe.csv")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    main(parser.parse_args())