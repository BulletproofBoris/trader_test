import argparse
import os
import sys
import pandas as pd
import numpy as np
from numba import njit
import traceback
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import json

BASE_DIR = Path(__file__).resolve().parent.parent

@njit
def _get_max_min_returns(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, horizon: int):
    n = len(closes)
    valid_len = max(0, n - horizon)
    max_rets = np.zeros(valid_len)
    min_rets = np.zeros(valid_len)
    for i in range(valid_len):
        c = closes[i]
        if c == 0 or np.isnan(c): continue
        future_highs = highs[i+1 : i+horizon+1]
        future_lows = lows[i+1 : i+horizon+1]
        max_rets[i] = (np.max(future_highs) - c) / c
        min_rets[i] = (np.min(future_lows) - c) / c
    return max_rets, min_rets

@njit
def _apply_triple_barrier(prices: np.ndarray, highs: np.ndarray, lows: np.ndarray, horizon: int, tp_factor: float, sl_factor: float) -> np.ndarray:
    n_rows = len(prices)
    labels = np.full(n_rows, 0.0) 
    for i in range(n_rows - 1):
        entry_price = prices[i]
        if entry_price == 0 or np.isnan(entry_price): continue
        upper_barrier = entry_price * tp_factor
        lower_barrier = entry_price * sl_factor
        path_end_idx = min(i + horizon, n_rows - 1)
        for j in range(i + 1, path_end_idx + 1):
            if highs[j] >= upper_barrier:
                labels[i] = 1.0 
                break 
            if lows[j] <= lower_barrier:
                labels[i] = 2.0 
                break
    return labels

def process_file_labeling(file_path: Path, output_dir: Path, horizon: int, tp: float, sl: float):
    try:
        df = pd.read_csv(file_path)
        required_cols = ['close', 'high', 'low']
        if df.empty or not all(col in df.columns for col in required_cols):
            return f"Пропущен (нет OHLC): {file_path.name}"
        labels = _apply_triple_barrier(df['close'].to_numpy(), df['high'].to_numpy(), df['low'].to_numpy(), horizon, 1 + tp / 100.0, 1 - sl / 100.0)
        df['label'] = labels
        df.to_csv(output_dir / file_path.name, index=False)
        return None
    except Exception:
        return f"--- Ошибка в {file_path.name} ---\n{traceback.format_exc()}"

def calculate_auto_parameters(files: list, horizon: int, target_percentile: int) -> tuple:
    print(f"📊 [АВТО-РЕЖИМ] Сканирую историю для расчета TP/SL (Горизонт: {horizon} баров)...")
    all_positive, all_negative = [], []
    for file_path in tqdm(files, desc="Анализ волатильности"):
        df = pd.read_csv(file_path, usecols=['close', 'high', 'low'])
        max_r, min_r = _get_max_min_returns(df['close'].to_numpy(), df['high'].to_numpy(), df['low'].to_numpy(), horizon)
        all_positive.extend(max_r[max_r > 0])
        all_negative.extend(min_r[min_r < 0])
    tp_auto = np.percentile(all_positive, target_percentile) * 100
    sl_auto = np.percentile(np.abs(all_negative), target_percentile) * 100
    print(f"📈 Авто TP/SL: {tp_auto:.2f}% / {sl_auto:.2f}%")
    return round(tp_auto, 2), round(sl_auto, 2)

def main(args):
    EXP_DIR = BASE_DIR / "experiments" / args.exp_name
    LABELED_DIR = EXP_DIR / "labels"
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    
    PROCESSED_DIR = BASE_DIR / "data" / "processed" / args.timeframe
    source_files = list(PROCESSED_DIR.glob("*_processed.csv"))
    if not source_files:
        print(f"❌ В {PROCESSED_DIR} не найдено файлов."); return

    final_tp, final_sl = args.tp, args.sl
    if args.auto:
        final_tp, final_sl = calculate_auto_parameters(source_files, args.horizon, args.percentile)

    exp_config = {
        "experiment_name": args.exp_name,
        "timeframe": args.timeframe,
        "lookback_bars": args.lookback,
        "horizon_bars": args.horizon,
        "take_profit_pct": final_tp,
        "stop_loss_pct": final_sl
    }
    with open(EXP_DIR / "exp_config.json", 'w', encoding='utf-8') as f:
        json.dump(exp_config, f, indent=4, ensure_ascii=False)

    print(f"\n🚀 Начинаю разметку {len(source_files)} файлов в {args.workers} потоков...")
    worker_func = partial(process_file_labeling, output_dir=LABELED_DIR, horizon=args.horizon, tp=final_tp, sl=final_sl)
    
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap(worker_func, source_files), total=len(source_files), desc="Разметка"))
    
    errors = [res for res in results if res is not None]
    if errors:
        for e in errors: print(e)
    else:
        print(f"🎉 Все датасеты успешно размечены и сохранены в {LABELED_DIR.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--timeframe', type=str, default="1d")
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--tp', type=float, default=5.0)
    parser.add_argument('--sl', type=float, default=3.0)
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--percentile', type=int, default=75)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    args = parser.parse_args()
    main(args)