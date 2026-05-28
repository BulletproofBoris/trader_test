import os
import argparse
import json
import numpy as np
import pandas as pd
import itertools
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, log_loss

# Отключаем спам TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def create_sequences(df, feature_cols, lookback):
    X, y, dates, tickers = [], [], [], []
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('datetime')
        features = group[feature_cols].values.astype(np.float32)
        labels = (group['label'].values + 1).astype(int)
        dt = group['datetime'].values
        tk = group['ticker'].values
        
        for i in range(len(features) - lookback):
            X.append(features[i : i + lookback])
            y.append(labels[i + lookback - 1])
            dates.append(dt[i + lookback - 1])
            tickers.append(tk[i + lookback - 1])
            
    return np.array(X), np.array(y), np.array(dates), np.array(tickers)

def load_top_models(models_dir, top_n=10):
    models_info = []
    for json_file in Path(models_dir).glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                model_name = meta.get("model_name")
                val_loss = meta.get("metrics", {}).get("val_loss", float('inf'))
                
                model_path = Path(models_dir) / model_name
                if model_path.exists():
                    models_info.append({"path": str(model_path), "loss": val_loss, "meta": meta})
        except Exception as e:
            pass
            
    models_info.sort(key=lambda x: x["loss"])
    return models_info[:top_n]

def main(args):
    dataset_dir = Path(args.dataset_dir)
    fold_dir = dataset_dir / args.fold
    models_dir = fold_dir / "models"
    artifacts_dir = fold_dir / "artifacts"
    val_parquet = fold_dir / "data" / "val" / "ml_data.parquet"
    
    print(f"🚀 Запуск КОМБИНАТОРНОГО Ансамблевого Анализатора ({args.fold})")
    
    with open(dataset_dir / "metadata.json", 'r', encoding='utf-8') as f:
        lookback = json.load(f)["parameters"]["lookback"]
        
    with open(artifacts_dir / "features_selected.json", 'r', encoding='utf-8') as f:
        feature_cols = json.load(f).get("feature_order", [])
        
    print(f"📦 Загрузка валидационных данных...")
    df_val = pd.read_parquet(val_parquet)
    X_val, y_val, dates_val, tickers_val = create_sequences(df_val, feature_cols, lookback)
    
    # Защита от экспоненциального взрыва
    max_k = min(args.max_k, 12) 
    top_models_info = load_top_models(models_dir, top_n=max_k)
    actual_k = len(top_models_info)
    
    if actual_k == 0:
        print("❌ Модели не найдены!")
        return
        
    print(f"\n🏆 Найдено {actual_k} элитных моделей. Генерация предиктов...")
    
    all_probs = []
    for i, info in enumerate(top_models_info, 1):
        print(f"  ⏳ Модель {i} (Indiv. Loss: {info['loss']:.4f})...")
        model = tf.keras.models.load_model(info['path'], compile=False)
        probs = model.predict(X_val, batch_size=2048, verbose=0)
        all_probs.append(probs)
        del model
        tf.keras.backend.clear_session()
        
    print(f"\n🤝 Просчет всех возможных связок ({(2**actual_k) - 1} комбинаций)...")
    results = []
    
    # Полный перебор комбинаций от 1 до N моделей
    for k in range(1, actual_k + 1):
        for combo in itertools.combinations(range(actual_k), k):
            combo_probs = [all_probs[i] for i in combo]
            ensemble_probs = np.mean(combo_probs, axis=0)
            
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            acc = accuracy_score(y_val, ensemble_preds)
            loss = log_loss(y_val, ensemble_probs)
            
            # Названия моделей (от 1 до N)
            combo_names = "+".join([str(i+1) for i in combo])
            
            results.append({
                "combo": combo_names,
                "k": k,
                "accuracy": acc,
                "log_loss": loss,
                "preds": ensemble_preds
            })

    # Сортируем результаты по Log Loss
    results.sort(key=lambda x: x['log_loss'])
    
    print("\n" + "="*65)
    print(f"{'Комбинация моделей':<25} | {'Кол-во (K)':<12} | {'Acc (%)':<10} | {'Log Loss':<10}")
    print("-" * 65)
    
    # Выводим Топ-15 лучших комбинаций
    for res in results[:15]:
        marker = "⭐ БЕСТСЕЛЛЕР" if res == results[0] else ""
        print(f"[{res['combo']:<23}] | K={res['k']:<10} | {res['accuracy']*100:<10.2f} | {res['log_loss']:<10.4f} {marker}")
    print("="*65)
    
    best_result = results[0]
    print(f"\n✅ Оптимальный альянс: Модели [{best_result['combo']}]")
    print("Детальный отчет для лучшей комбинации:")
    print(classification_report(y_val, best_result['preds'], target_names=['SL (-1)', 'Hold (0)', 'TP (+1)']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Автоматический поиск лучших комбинаций ансамбля")
    parser.add_argument("--dataset_dir", type=str, default="data/processed/2000_2026_1d_6_1")
    parser.add_argument("--fold", type=str, default="fold_2010")
    parser.add_argument("--max_k", type=int, default=10, help="Сколько топ-моделей взять для перебора (Макс 12)")
    args = parser.parse_args()
    
    main(args)