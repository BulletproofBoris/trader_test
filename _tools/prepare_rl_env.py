import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BASE_DIR = Path(__file__).resolve().parent.parent

def load_best_model(exp_name: str):
    models_dir = BASE_DIR / "experiments" / exp_name / "models"
    models = list(models_dir.glob("trading_bot_best_acc_*.keras"))
    models.sort(key=lambda x: float(x.stem.split('_')[4]), reverse=True)
    best_model_path = models[0]
    print(f"✅ Загружена модель {exp_name}: {best_model_path.name}")
    return tf.keras.models.load_model(best_model_path)

def get_predictions(exp_name: str, model):
    print(f"🧠 Генерация прогнозов от {exp_name}...")
    exp_dir = BASE_DIR / "experiments" / exp_name
    
    with open(exp_dir / "exp_config.json", 'r') as f:
        config = json.load(f)
    lookback = config["lookback_bars"]
    full_features = config.get("feature_order", [])
    
    selected_features_path = exp_dir / "dataset" / "features_selected.json"
    reduced_features = []
    if selected_features_path.exists():
        with open(selected_features_path, 'r') as f:
            reduced_features = json.load(f)["feature_order"]
            
    expected_feature_count = model.input_shape[-1]
    features = reduced_features if expected_feature_count == len(reduced_features) else full_features

    # Берем ТОЛЬКО тестовые данные (2024-2026), которых LSTM никогда не видел
    df = pd.read_csv(exp_dir / "dataset" / "test_scaled.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    all_results = []
    for ticker, group in tqdm(df.groupby('ticker'), desc=f"Прогнозы {exp_name}"):
        group = group.sort_values('datetime').reset_index(drop=True)
        if len(group) <= lookback: continue
            
        feature_data = group[features].values
        dates = group['datetime'].values
        closes = group['close'].values 
        
        X = [feature_data[i : i + lookback] for i in range(len(feature_data) - lookback)]
        if not X: continue
            
        preds = model.predict(np.array(X), verbose=0, batch_size=2048)
        
        for i in range(len(preds)):
            all_results.append({
                'datetime': dates[i + lookback - 1],
                'ticker': ticker,
                'close': closes[i + lookback - 1], 
                f'{exp_name}_prob_flat': preds[i][0],
                f'{exp_name}_prob_tp': preds[i][1],
                f'{exp_name}_prob_sl': preds[i][2],
            })
            
    return pd.DataFrame(all_results)

def main(args):
    RL_DIR = BASE_DIR / "experiments" / "rl_trader"
    RL_DIR.mkdir(parents=True, exist_ok=True)
    
    model_fast = load_best_model(args.exp_fast)
    model_slow = load_best_model(args.exp_slow)
    
    df_fast = get_predictions(args.exp_fast, model_fast)
    df_slow = get_predictions(args.exp_slow, model_slow)
    
    print("\n🔗 Слияние прогнозов...")
    merged_df = pd.merge(df_fast, df_slow.drop(columns=['close']), on=['datetime', 'ticker'], how='inner')
    merged_df.sort_values(by=['ticker', 'datetime'], inplace=True)
    
    # Хронологический сплит!
    # Обучаем RL на 2024 и 2025 году
    train_rl = merged_df[merged_df['datetime'] < '2026-01-01']
    # Тестируем строго на 2026 годе
    test_rl = merged_df[merged_df['datetime'] >= '2026-01-01']
    
    train_rl.to_csv(RL_DIR / "rl_train_dataset.csv", index=False)
    test_rl.to_csv(RL_DIR / "rl_test_dataset.csv", index=False)
    
    print(f"✅ Датасеты готовы!")
    print(f"   RL Train (2024-2025): {len(train_rl)} строк")
    print(f"   RL Test  (2026):      {len(test_rl)} строк")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_fast', type=str, default="exp_30_5_1d")
    parser.add_argument('--exp_slow', type=str, default="exp_60_10_1d")
    args = parser.parse_args()
    main(args)