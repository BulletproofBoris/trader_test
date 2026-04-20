# _tools/fit_scaler.py
# -*- coding: utf-8 -*-
"""
Обучение Pipeline (Imputer + RobustScaler) на train-данных и сохранение «безопасного» скейлера.
Запуск:
    python -m _tools.fit_scaler --model_name EXP1
Опции:
    --train_path      (необязательно) путь к train_final.csv; по умолчанию берётся из final_for_model/<model_name>/
    --label_col       имя колонки с меткой (по умолчанию: label)
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import joblib

# --- Подготовка путей для импортов пакета ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from _core.paths import RESULTS_DIR, ensure_dirs  # PROJECT_ROOT не обязателен здесь

warnings.simplefilter("ignore", UserWarning)


def _resolve_paths(model_name: str, train_path: str | None) -> Tuple[Path, Path, Path, Path]:
    """
    Возвращает (train_csv, scalers_dir, scaler_pkl_target)
    """
    base = RESULTS_DIR / "final_for_model" / model_name
    if train_path:
        train_csv = Path(train_path)
    else:
        train_csv = base / "train_final.csv"
    scalers_dir = RESULTS_DIR / "scalers"
    scaler_pkl = scalers_dir / f"{model_name}_scaler.pkl"
    features_json = scalers_dir / f"{model_name}_features.json"
    return train_csv, scalers_dir, scaler_pkl, features_json


def _detect_feature_columns(df: pd.DataFrame, label_col: str) -> List[str]:
    bad_cols = {"datetime", "ticker", label_col}
    # Берём только числовые и НЕ служебные
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = [c for c in num_cols if c not in bad_cols]
    return feat_cols


def _quick_sanity_check(X: np.ndarray) -> Tuple[int, int]:
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    return n_nan, n_inf


def main(args: argparse.Namespace) -> None:
    ensure_dirs()

    train_csv, scalers_dir, scaler_pkl, features_json = _resolve_paths(args.model_name, args.train_path)

    if not train_csv.exists():
        raise FileNotFoundError(f"Не найден train csv: {train_csv}")

    print("—" * 60)
    print("🧰 Обучение скейлера (Imputer + RobustScaler)")
    print(f"Модель: {args.model_name}")
    print(f"Загрузка тренировочных данных из: {train_csv}")
    print("—" * 60)

    df = pd.read_csv(train_csv)
    if df.empty:
        raise ValueError("Файл train пустой.")

    label_col = args.label_col
    if label_col not in df.columns:
        # не падаем жёстко — бывает, что метка как-то иначе называется в этом этапе
        warnings.warn(f"Колонка с меткой '{label_col}' не найдена в train. Продолжаю без неё.")

    feat_cols = _detect_feature_columns(df, label_col=label_col)
    if not feat_cols:
        raise ValueError("Не найдены числовые признаки для обучения скейлера.")

    X_df = df[feat_cols]
    print(f"Признаков: {len(feat_cols)}, обучающих сэмплов: {X_df.shape[0]}")

    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0)))
    ])

    print("Обучение Pipeline (Imputer + RobustScaler)...")
    pipeline.fit(X_df)
    print("✅ Pipeline успешно обучен.")

    # --- 👇 ИСПРАВЛЕНИЕ: Используем X_df для sanity-проверки 👇 ---
    Xt = pipeline.transform(X_df.head(5000))
    n_nan, n_inf = _quick_sanity_check(Xt)
    if n_nan or n_inf:
        warnings.warn(f"После transform найдены NaN/Inf: NaN={n_nan}, Inf={n_inf}")

    # Сохраняем «безопасный» Pipeline и список признаков
    scalers_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, scaler_pkl)
    with open(features_json, "w", encoding="utf-8") as f:
        json.dump({"feature_order": feat_cols}, f, ensure_ascii=False, indent=2)

    print(f"✅ Абсолютно 'безопасный' Pipeline сохранен в: {scaler_pkl}")
    print(f"🗂  Порядок признаков сохранен в: {features_json}")
    print("Готово.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение скейлера по train_final.csv")
    parser.add_argument("--model_name", type=str, required=True, help="Имя модели/эксперимента (папка в final_for_model)")
    parser.add_argument("--train_path", type=str, default=None, help="Явный путь к train_final.csv (если нужно переопределить)")
    parser.add_argument("--label_col", type=str, default="label", help="Имя столбца метки")
    main(parser.parse_args())