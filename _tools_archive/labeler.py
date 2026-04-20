# _tools/labeler.py

import argparse
import pandas as pd
import numpy as np
from numba import njit
import traceback
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import os

# --- Автономное определение путей ---
try:
    PROJECT_DIR = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_DIR = Path.cwd()

RESULTS_DIR = PROJECT_DIR / "2_results"
FEATURES_DIR = RESULTS_DIR / "featuresets" 
FEATURES_DIR_PROCESSED = RESULTS_DIR / "features_processed" 
# --- Конец блока ---

@njit
def _apply_triple_barrier(
    prices: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    horizon: int, tp_factor: float, sl_factor: float
) -> np.ndarray:
    n_rows = len(prices)
    labels = np.full(n_rows, 0.0) 

    for i in range(n_rows - 1):
        entry_price = prices[i]
        if entry_price == 0: continue

        upper_barrier = entry_price * tp_factor
        lower_barrier = entry_price * sl_factor
        
        path_end_idx = min(i + horizon, n_rows - 1)

        for j in range(i + 1, path_end_idx + 1):
            if highs[j] >= upper_barrier:
                labels[i] = 1.0
                break 
            if lows[j] <= lower_barrier:
                labels[i] = -1.0
                break
    return labels

def process_file(file_path: Path, args: argparse.Namespace):
    """Читает, размечает, добавляет тикер и сохраняет один файл."""
    try:
        df = pd.read_csv(file_path)
        required_cols = ['close', 'high', 'low']
        if df.empty or not all(col in df.columns for col in required_cols):
            return f"Пропущен (нет OHLC): {file_path.name}"

        labels = _apply_triple_barrier(
            prices=df['close'].to_numpy(),
            highs=df['high'].to_numpy(),
            lows=df['low'].to_numpy(),
            horizon=args.horizon,
            tp_factor=1 + args.tp / 100.0,
            sl_factor=1 - args.sl / 100.0
        )
        
        df_labeled = df.copy()
        
        # --- 👇👇👇 ГЛАВНОЕ ИСПРАВЛЕНИЕ 👇👇👇 ---
        # Переименовываем колонку с 'target' на 'label'
        df_labeled['label'] = labels
        # --- 👆👆👆 КОНЕЦ ИСПРАВЛЕНИЯ 👆👆👆 ---
        
        ticker = file_path.stem.split('_')[0]
        df_labeled['ticker'] = ticker
        
        output_path = FEATURES_DIR_PROCESSED / file_path.name
        df_labeled.to_csv(output_path, index=False)
        
        return None
    except Exception:
        tb_str = traceback.format_exc()
        return f"--- Ошибка в файле {file_path.name} ---\n{tb_str}"

def main(args):
    FEATURES_DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    print("--- Шаг 2.5: Разметка данных методом 'Тройной преграды' ---")
    print(f"Параметры: Горизонт={args.horizon} дней, TakeProfit={args.tp}%, StopLoss={args.sl}%")
    
    source_files = list(FEATURES_DIR.glob("*.csv"))
    if not source_files:
        print(f"❌ В папке {FEATURES_DIR} не найдено файлов."); return

    print(f"Найдено {len(source_files)} файлов для обработки. Запускаю в {args.workers} потоков...")

    worker_func = partial(process_file, args=args)
    
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap(worker_func, source_files), total=len(source_files), desc="Разметка файлов"))
    
    errors = [res for res in results if res is not None]
    
    print(f"\n✅ Процесс разметки завершен.")
    if errors:
        print(f"❗️ Обнаружено ошибок/пропусков: {len(errors)} из {len(source_files)}")
    else:
        print("🎉 Все файлы обработаны без ошибок.")
        
    print(f"Обработанные файлы сохранены в: {FEATURES_DIR_PROCESSED}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Разметка данных методом тройного барьера.")
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--tp', type=float, default=7.0, help="Take Profit в процентах.")
    parser.add_argument('--sl', type=float, default=4.0, help="Stop Loss в процентах.")
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help="Количество потоков.")
    
    args = parser.parse_args()
    main(args)