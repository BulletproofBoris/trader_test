import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent

def prepare_data(exp_name: str, split_date: str):
    EXP_DIR = BASE_DIR / "experiments" / exp_name
    LABELED_DIR = EXP_DIR / "labels"
    ML_DATA_DIR = EXP_DIR / "dataset"
    ML_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    config_path = EXP_DIR / "exp_config.json"
    if not config_path.exists():
        print(f"❌ Паспорт не найден: {config_path}"); return
        
    with open(config_path, 'r', encoding='utf-8') as f:
        exp_config = json.load(f)
    
    exp_config["split_date"] = split_date
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(exp_config, f, indent=4, ensure_ascii=False)

    print(f"🚀 Сборка, Сплит и Масштабирование для: {exp_name}")
    csv_files = list(LABELED_DIR.glob("*_processed.csv"))
    
    df_list = []
    for file in tqdm(csv_files, desc="Чтение"):
        df_list.append(pd.read_csv(file))
        
    full_df = pd.concat(df_list, ignore_index=True)
    full_df['datetime'] = pd.to_datetime(full_df['datetime'])
    full_df.sort_values(by=['datetime', 'ticker'], inplace=True)
    
    split_timestamp = pd.to_datetime(split_date)
    train_df = full_df[full_df['datetime'] < split_timestamp].copy()
    test_df = full_df[full_df['datetime'] >= split_timestamp].copy()
    
    print(f"\n✂️ Хронологический сплит по дате: {split_date}")
    print(f"📈 Train: {len(train_df):,} | Test: {len(test_df):,}")

    exclude_cols = {'datetime', 'ticker', 'label'}
    numeric_cols = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c])]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    print(f"🧰 Обучение скейлера на {len(feature_cols)} признаках...")
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0)))
    ])
    
    pipeline.fit(train_df[feature_cols])
    
    print("🔄 Применение скейлера...")
    train_scaled_features = pipeline.transform(train_df[feature_cols])
    test_scaled_features = pipeline.transform(test_df[feature_cols])
    
    for i, col in enumerate(feature_cols):
        train_df[col] = train_scaled_features[:, i]
        test_df[col] = test_scaled_features[:, i]

    print("💾 Сохранение готовых данных...")
    train_df.to_csv(ML_DATA_DIR / "train_scaled.csv", index=False)
    test_df.to_csv(ML_DATA_DIR / "test_scaled.csv", index=False)
    joblib.dump(pipeline, ML_DATA_DIR / "scaler.pkl")
    
    print(f"✅ Финальные данные лежат в {ML_DATA_DIR.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--split_date', type=str, default="2024-01-01")
    args = parser.parse_args()
    prepare_data(args.exp_name, args.split_date)