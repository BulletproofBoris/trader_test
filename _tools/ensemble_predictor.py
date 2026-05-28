import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, log_loss

# Отключаем спам TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def create_sequences(df, feature_cols, lookback):
    """Нарезает датасет на 3D-тензоры (Samples, Lookback, Features)"""
    X, y, dates, tickers = [], [], [], []
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('datetime')
        features = group[feature_cols].values.astype(np.float32)
        labels = (group['label'].values + 1).astype(int) # Сдвиг [-1, 0, 1] -> [0, 1, 2]
        dt = group['datetime'].values
        tk = group['ticker'].values
        
        for i in range(len(features) - lookback):
            X.append(features[i : i + lookback])
            y.append(labels[i + lookback - 1])
            dates.append(dt[i + lookback - 1])
            tickers.append(tk[i + lookback - 1])
            
    return np.array(X), np.array(y), np.array(dates), np.array(tickers)

def load_top_models(models_dir, top_n=3):
    """Ищет N лучших моделей на основе их JSON-метаданных"""
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
            print(f"⚠️ Ошибка чтения {json_file.name}: {e}")
            
    # Сортируем по возрастанию Loss и берем Топ-N
    models_info.sort(key=lambda x: x["loss"])
    return models_info[:top_n]

def main(args):
    dataset_dir = Path(args.dataset_dir)
    fold_dir = dataset_dir / args.fold
    models_dir = fold_dir / "models"
    artifacts_dir = fold_dir / "artifacts"
    val_parquet = fold_dir / "data" / "val" / "ml_data.parquet"
    
    print(f"🚀 Запуск Ансамблевого Предиктора для {args.fold}")
    
    # 1. Загрузка метаданных и фичей
    with open(dataset_dir / "metadata.json", 'r', encoding='utf-8') as f:
        lookback = json.load(f)["parameters"]["lookback"]
        
    with open(artifacts_dir / "features_selected.json", 'r', encoding='utf-8') as f:
        feature_cols = json.load(f).get("feature_order", [])
        
    # 2. Загрузка данных
    print(f"📦 Загрузка валидационных данных...")
    df_val = pd.read_parquet(val_parquet)
    X_val, y_val, dates_val, tickers_val = create_sequences(df_val, feature_cols, lookback)
    print(f"✅ Тензор сформирован: {X_val.shape}")
    
    # 3. Поиск Топ-3 моделей
    top_models_info = load_top_models(models_dir, top_n=args.top_k)
    if not top_models_info:
        print("❌ Модели не найдены!")
        return
        
    print(f"\n🏆 Найдено {len(top_models_info)} элитных моделей для ансамбля:")
    loaded_models = []
    for i, info in enumerate(top_models_info, 1):
        print(f"  {i}. {Path(info['path']).name} (Loss: {info['loss']:.4f})")
        # Загружаем модель в память
        model = tf.keras.models.load_model(info['path'], compile=False)
        loaded_models.append(model)
        
    # 4. Получение предсказаний
    print("\n🧠 Генерация вероятностей (Softmax)...")
    all_probs = []
    
    for i, model in enumerate(loaded_models, 1):
        # Делаем предикт батчами, чтобы не выбить OOM
        probs = model.predict(X_val, batch_size=2048, verbose=0)
        
        # Считаем метрики одиночной модели
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_val, preds)
        loss = log_loss(y_val, probs)
        print(f"  👉 Модель {i} | Acc: {acc*100:.2f}% | Loss: {loss:.4f}")
        
        all_probs.append(probs)
        
    # 5. АНСАМБЛИРОВАНИЕ (Усреднение вероятностей)
    print("\n🤝 Слияние предсказаний (Ensembling)...")
    # all_probs имеет форму (Num_Models, Samples, Classes)
    # Усредняем по оси 0 (по моделям)
    ensemble_probs = np.mean(all_probs, axis=0)
    
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_acc = accuracy_score(y_val, ensemble_preds)
    ensemble_loss = log_loss(y_val, ensemble_probs)
    
    print("="*50)
    print(f"🌟 РЕЗУЛЬТАТЫ АНСАМБЛЯ ({len(loaded_models)} моделей):")
    print(f"   Ensemble Accuracy : {ensemble_acc*100:.2f}%")
    print(f"   Ensemble Log Loss : {ensemble_loss:.4f}")
    print("="*50)
    
    # Детальный отчет
    print("\nДетальный отчет Ансамбля по классам:")
    target_names = ['SL (-1)', 'Hold (0)', 'TP (+1)']
    print(classification_report(y_val, ensemble_preds, target_names=target_names))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/processed/2000_2026_1d_10_5")
    parser.add_argument("--fold", type=str, default="fold_2010")
    parser.add_argument("--top_k", type=int, default=3, help="Количество лучших моделей для ансамбля")
    args = parser.parse_args()
    
    main(args)