import os
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def add_features(df):
    """Генерация технических признаков"""
    df = df.sort_values(by=['ticker', 'datetime'])
    
    # Базовые фичи
    df['ret_1d'] = df.groupby('ticker')['close'].pct_change()
    df['ret_5d'] = df.groupby('ticker')['close'].pct_change(5)
    
    # Трендовые
    df['close_ma_10'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(10).mean())
    df['close_ma_50'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(50).mean())
    df['dist_to_ma10'] = df['close'] / (df['close_ma_10'] + 1e-8) - 1.0
    
    # Объемные
    df['vol_ma_5'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(5).mean())
    df['vol_ratio'] = df['volume'] / (df['vol_ma_5'] + 1e-8)
    
    return df.dropna().copy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--phase', type=str, required=True, choices=['train', 'val'])
    parser.add_argument('--artifacts_dir', type=str, required=True)
    
    # Аргументы для фильтрации утечек
    parser.add_argument('--start_date', type=str, help="Реальная дата начала фазы (без буфера)")
    
    # Заглушки
    parser.add_argument('--timeframe', type=str)
    parser.add_argument('--lookback', type=int)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--tp', type=float)
    parser.add_argument('--sl', type=float)
    parser.add_argument('--percentile', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--auto', action='store_true')
    
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"⚙️ [{args.phase.upper()}] Генерация признаков...")
    df = add_features(df)
    
    exclude_cols = ['datetime', 'ticker', 'target_tp', 'target_sl', 'target_return', 'label', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # --- ВИНЗОРИЗАЦИЯ (Защита от выбросов) ---
    print("✂️ Winsorization (0.1% - 99.9%)...")
    for col in feature_cols:
        lower = df[col].quantile(0.001)
        upper = df[col].quantile(0.999)
        df[col] = df[col].clip(lower=lower, upper=upper)

    scaler_path = Path(args.artifacts_dir) / "scaler_features.pkl"
    scaler = StandardScaler()

    if args.phase == 'train':
        print(f"📈 Обучение Scaler...")
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        joblib.dump(scaler, scaler_path)
    elif args.phase == 'val':
        if not scaler_path.exists():
            raise FileNotFoundError("Scaler pkl not found in artifacts!")
        print(f"📉 Применение Scaler...")
        loaded_scaler = joblib.load(scaler_path)
        df[feature_cols] = loaded_scaler.transform(df[feature_cols])
        
        # --- ФИКС УТЕЧКИ: Обрезаем буферные строки ---
        if args.start_date:
            original_len = len(df)
            df = df[df['datetime'] >= pd.to_datetime(args.start_date)]
            print(f"🛡️ Удалено {original_len - len(df)} строк буфера (Data Leakage Protection)")

    # Приведение типов и сохранение
    for col in feature_cols:
        df[col] = df[col].astype(np.float32)
        
    df.to_parquet(args.output, engine='pyarrow', index=False)
    print(f"💾 Сохранено в {Path(args.output).name}")

if __name__ == "__main__":
    main()