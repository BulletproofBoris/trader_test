import os
import json
import argparse
import subprocess
import hashlib
import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta
from _tools.data_cleaner import clean_and_adjust_data

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def run_step(cmd, step_name):
    print(f"\n🔄 [{step_name}] Запуск...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка на этапе {step_name}. Код: {e.returncode}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Мастер-генератор интервальных датасетов (Walk-Forward)")
    
    # Параметры разметки и фичей
    parser.add_argument('--timeframe', type=str, default="1d")
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--tp', type=float, default=5.0)
    parser.add_argument('--sl', type=float, default=3.0)
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--percentile', type=int, default=75)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    
    # Параметры Feature Selection
    parser.add_argument('--cum_threshold', type=float, default=0.99)
    parser.add_argument('--corr_threshold', type=float, default=0.85)

    # Параметры Walk-Forward
    parser.add_argument('--init_split', type=str, required=True)
    parser.add_argument('--val_interval', type=int, required=True)
    parser.add_argument('--split_interval', type=int, required=True)
    parser.add_argument('--endpoint', type=str, required=True)

    args = parser.parse_args()

    BASE_DIR = Path("/home/restorator/trader_test")
    TOOLS_DIR = BASE_DIR / "_tools"
    RAW_DIR = BASE_DIR / "data" / "raw" / args.timeframe
    PROCESSED_BASE = BASE_DIR / "data" / "processed"

    # --- СБОРКА И ОЧИСТКА ДАННЫХ ---
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv') and not f.endswith('_meta.csv')]
    df_list = []
    for file in csv_files:
        try:
            df_temp = pd.read_csv(RAW_DIR / file)
            df_temp.columns = [c.strip().lower() for c in df_temp.columns]
            date_col = next((c for c in ['date', 'datetime', 'tradedate', 'time'] if c in df_temp.columns), None)
            if not date_col: continue
            df_temp.rename(columns={date_col: 'datetime'}, inplace=True)
            if 'ticker' not in df_temp.columns:
                raw_name = file.split('_1D')[0].split('_MAX')[0].split('.')[0]
                df_temp['ticker'] = raw_name.replace("_", "@", 1) if ("MACRO" not in raw_name and "_" in raw_name) else raw_name
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in df_temp.columns: df_temp[col] = df_temp['close'] if 'close' in df_temp.columns else 0.0
            df_list.append(df_temp[['datetime', 'ticker', 'open', 'high', 'low', 'close', 'volume']])
        except: continue

    df_combined = pd.concat(df_list, ignore_index=True)
    df_combined['datetime'] = pd.to_datetime(df_combined['datetime'])
    df_combined = clean_and_adjust_data(df_combined.sort_values(['datetime', 'ticker']))

    dataset_name = f"{df_combined['datetime'].min().year}_{df_combined['datetime'].max().year}_{args.timeframe}"
    DATASET_DIR = PROCESSED_BASE / dataset_name
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    CACHE_RAW_FILE = DATASET_DIR / "raw_combined.csv"
    df_combined.to_csv(CACHE_RAW_FILE, index=False)
    
    with open(DATASET_DIR / "metadata.json", "w") as f:
        json.dump({"tickers": sorted(df_combined['ticker'].unique().tolist()), "parameters": vars(args)}, f, indent=4)

    # --- ЦИКЛ WALK-FORWARD ---
    base_script_args = ["--timeframe", args.timeframe, "--lookback", str(args.lookback), "--horizon", str(args.horizon), "--tp", str(args.tp), "--sl", str(args.sl), "--percentile", str(args.percentile), "--workers", str(args.workers)]
    if args.auto: base_script_args.append("--auto")

    current_split = pd.to_datetime(args.init_split)
    endpoint = pd.to_datetime(args.endpoint)
    min_date = df_combined['datetime'].min()

    while current_split <= endpoint:
        fold_name = f"fold_{current_split.year}"
        FOLD_DIR = DATASET_DIR / fold_name
        ARTIFACTS_DIR = FOLD_DIR / "artifacts" # ОПРЕДЕЛЯЕМ ЗДЕСЬ
        
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(FOLD_DIR / "models", exist_ok=True)

        train_end = (current_split - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        val_start = current_split.strftime('%Y-%m-%d')
        val_end = (current_split + relativedelta(years=args.val_interval) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        for phase, p_start, p_end in [("train", min_date.strftime('%Y-%m-%d'), train_end), ("val", val_start, val_end)]:
            PHASE_DIR = FOLD_DIR / "data" / phase
            os.makedirs(PHASE_DIR, exist_ok=True)
            
            fetch_start = p_start if phase == "train" else (pd.to_datetime(p_start) - pd.Timedelta(days=args.lookback * 2)).strftime('%Y-%m-%d')
            out_ds, out_lbl, out_ml = PHASE_DIR/"dataset.csv", PHASE_DIR/"labels.csv", PHASE_DIR/"ml_data.parquet"

            run_step(["python", str(TOOLS_DIR/"build_dataset.py"), "--input", str(CACHE_RAW_FILE), "--output", str(out_ds), "--start_date", fetch_start, "--end_date", p_end], f"{fold_name}/{phase} - Build")
            run_step(["python", str(TOOLS_DIR/"create_labels.py"), "--input", str(out_ds), "--output", str(out_lbl)] + base_script_args, f"{fold_name}/{phase} - Labels")
            run_step(["python", str(TOOLS_DIR/"prepare_ml_data.py"), "--input", str(out_lbl), "--output", str(out_ml), "--phase", phase, "--artifacts_dir", str(ARTIFACTS_DIR), "--start_date", p_start] + base_script_args, f"{fold_name}/{phase} - Features")

        # --- ФИЧИ: ОТБОР (Внутри цикла фолда, после генерации train/val) ---
        train_parquet = FOLD_DIR / "data/train/ml_data.parquet"
        features_json = ARTIFACTS_DIR / "features_selected.json"
        features_plot = FOLD_DIR / "models/dynamic_feature_importance.png"
        
        run_step([
            "python", str(TOOLS_DIR / "feature_selector.py"),
            "--input", str(train_parquet),
            "--out_json", str(features_json),
            "--out_plot", str(features_plot),
            "--cum_threshold", str(args.cum_threshold),
            "--corr_threshold", str(args.corr_threshold)
        ], f"{fold_name} - Feature Selection")

        current_split += relativedelta(years=args.split_interval)
    print("\n✅ ПАЙПЛАЙН ПОЛНОСТЬЮ ЗАВЕРШЕН!")

if __name__ == "__main__":
    main()