# _tools/clean_timeseries.py
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob
import os
from typing import List, Tuple

# --- Настройка путей ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "2_results"
# ---

REQUIRED_COLS = ["datetime", "open", "high", "low", "close", "volume"]

# --- ИЗМЕНЕННАЯ ФУНКЦИЯ: Теперь возвращает лог изменений ---
def remove_outliers(df: pd.DataFrame, abs_daily_threshold: float, sma_window: int, sma_multiplier: float) -> Tuple[pd.DataFrame, List[str]]:
    if df.empty or len(df) < sma_window:
        return df, []

    log_messages = []
    df_clean = df.copy()

    # --- Фильтр 1: Экстремальные однодневные изменения ---
    df_clean['daily_return'] = df_clean['close'].pct_change()
    returns_for_masking = df_clean['daily_return'].fillna(0)
    abs_mask = (returns_for_masking < abs_daily_threshold) & (returns_for_masking > -0.95)
    
    abs_deleted_df = df_clean[~abs_mask]
    if not abs_deleted_df.empty:
        log_messages.append(f"    (i) Фильтр 1 (однодневный > {abs_daily_threshold*100:.0f}%): удалено {len(abs_deleted_df)} строк.")
        for dt in abs_deleted_df['datetime']:
            log_messages.append(f"        - {pd.to_datetime(dt).date()}")
    
    df_clean = df_clean[abs_mask]

    # --- Фильтр 2: Отклонение от скользящего среднего ---
    if sma_multiplier > 0 and not df_clean.empty:
        df_clean['sma'] = df_clean['close'].rolling(window=sma_window, min_periods=1).mean().bfill()
        sma_mask = df_clean['close'] < (df_clean['sma'] * sma_multiplier)
        
        sma_deleted_df = df_clean[~sma_mask]
        if not sma_deleted_df.empty:
            log_messages.append(f"    (i) Фильтр 2 (цена > SMA*{sma_multiplier}): удалено {len(sma_deleted_df)} строк.")
            for dt in sma_deleted_df['datetime']:
                log_messages.append(f"        - {pd.to_datetime(dt).date()}")
        
        df_clean = df_clean[sma_mask]

    cols_to_drop = [col for col in ['daily_return', 'sma'] if col in df_clean.columns]
    df_final = df_clean.drop(columns=cols_to_drop)
    
    return df_final, log_messages
# -----------------------------------------------------------------

def read_ohlcv(path: str) -> pd.DataFrame | None:
    # ... (код без изменений) ...
    if not os.path.exists(path): return None
    try: df = pd.read_csv(path)
    except Exception: return None
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "datetime" not in df.columns:
        for alt in ("date", "timestamp", "time", "tradedate"):
            if alt in df.columns:
                df = df.rename(columns={alt: "datetime"}); break
    for col in REQUIRED_COLS:
        if col not in df.columns: df[col] = np.nan
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False, errors='coerce')
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    return df[REQUIRED_COLS].copy()

def drop_blank_ohlcv_rows(df: pd.DataFrame) -> pd.DataFrame:
    # ... (код без изменений) ...
    prices_nan = df[["open", "high", "low", "close"]].isna().all(axis=1)
    zero_vol = df["volume"].fillna(0).eq(0)
    return df.loc[~(prices_nan & zero_vol)].copy()

def split_into_segments(df: pd.DataFrame, *, gap_ok_days: int = 3, gap_buffer_days: int = 1, min_seg_len: int = 5) -> pd.DataFrame:
    # ... (код без изменений) ...
    if df.empty: return df
    df = df.reset_index(drop=True)
    dates = pd.to_datetime(df["datetime"]).dt.normalize()
    diffs = dates.diff().dt.days.fillna(0).astype(int)
    big_gap_idx = df.index[diffs > gap_ok_days].to_list()
    if not big_gap_idx: return df
    bounds: List[Tuple[int, int]] = []
    start_idx = df.index[0]
    for i in big_gap_idx:
        end_idx = i - 1
        bounds.append((start_idx, end_idx))
        start_idx = i
    bounds.append((start_idx, df.index[-1]))
    kept_parts: List[pd.DataFrame] = []
    for k, (s, e) in enumerate(bounds):
        part = df.iloc[s:e+1]
        trim_left = gap_buffer_days if k > 0 else 0
        trim_right = gap_buffer_days if k < len(bounds) - 1 else 0
        if trim_left > 0 and len(part) > trim_left: part = part.iloc[trim_left:]
        if trim_right > 0 and len(part) > trim_right: part = part.iloc[:-trim_right]
        if len(part) >= min_seg_len: kept_parts.append(part)
    if not kept_parts: return df.iloc[0:0].copy()
    out = pd.concat(kept_parts, ignore_index=True)
    return out

# --- ИЗМЕНЕННАЯ ФУНКЦИЯ: Теперь возвращает лог ---
def clean_file(path: str, args: argparse.Namespace) -> List[str]:
    df = read_ohlcv(path)
    if df is None or df.empty:
        return []

    log_messages = []
    df = drop_blank_ohlcv_rows(df)

    if args.remove_outliers:
        df, outlier_logs = remove_outliers(df, args.outlier_abs_threshold, args.outlier_sma_window, args.outlier_sma_multiplier)
        log_messages.extend(outlier_logs)
    
    df = split_into_segments(df, gap_ok_days=args.gap_ok, gap_buffer_days=args.gap_buffer, min_seg_len=args.min_seg_len)
    
    if not df.empty:
        df.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
        
    return log_messages

def main():
    ap = argparse.ArgumentParser(description="Очистка временных рядов OHLCV.")
    # ... (аргументы без изменений) ...
    ap.add_argument("--data-dir", default="1_data")
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--gap-ok", type=int, default=3)
    ap.add_argument("--gap-buffer", type=int, default=2)
    ap.add_argument("--min-seg-len", type=int, default=5)
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--remove-outliers", action="store_true")
    ap.add_argument("--outlier-abs-threshold", type=float, default=2.0)
    ap.add_argument("--outlier-sma-window", type=int, default=20)
    ap.add_argument("--outlier-sma-multiplier", type=float, default=3.0)
    args = ap.parse_args()

    files = sorted(glob(str(Path(args.data_dir) / args.pattern)))
    
    # --- ИЗМЕНЕННЫЙ ЦИКЛ: Используем tqdm.write для "чистого" вывода ---
    with tqdm(files, desc="Cleaning OHLCV") as pbar:
        for path in pbar:
            log_messages = clean_file(path, args)
            
            # Если для файла были сообщения (т.е. были удаления), выводим отчет
            if log_messages:
                pbar.write(f"\n--- 🧰 Processing: {Path(path).name} ---")
                for msg in log_messages:
                    pbar.write(msg)
    
    print(f"\n[OK] Очистка завершена.")

if __name__ == "__main__":
    main()