import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

def parse_tfrecord_fn(example, lookback, n_features):
    feature_description = {
        'sequence': tf.io.FixedLenFeature([], tf.string), 
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed = tf.io.parse_single_example(example, feature_description)
    # Исправлено 'features' на 'sequence'
    sequence = tf.io.parse_tensor(parsed['sequence'], out_type=tf.float32)
    sequence.set_shape([lookback, n_features])
    return sequence

def get_top_n_models_in_fold(fold_dir, top_n=3):
    """Ищет топ-N моделей с минимальным val_loss в папке models"""
    models_dir = fold_dir / "models"
    if not models_dir.exists(): return []
    
    valid_models = []
    for json_path in models_dir.glob("*.json"):
        try:
            with open(json_path, 'r') as f:
                meta = json.load(f)
                val_loss = meta.get("metrics", {}).get("val_loss")
                if val_loss is not None:
                    model_path = models_dir / meta["model_name"]
                    if model_path.exists():
                        valid_models.append((model_path, meta, val_loss))
        except Exception:
            pass
            
    valid_models.sort(key=lambda x: x[2])
    return valid_models[:top_n]

def main():
    base_dataset_dir = Path("data/processed/2000_2026_1d_60_10")
    output_dir = Path("data/processed/2000_2026_1d/rl_env")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_parquet = output_dir / "environment_data.parquet"

    print(f"🚀 Генерация 3-х последовательностей (Multi-Model) для: {base_dataset_dir.name}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

    folds = sorted([d for d in base_dataset_dir.glob("fold_*") if d.is_dir()])
    if not folds:
        print("❌ Фолды не найдены.")
        return

    all_predictions = []

    for fold_dir in folds:
        print(f"\n📂 Обработка фолда: {fold_dir.name}")
        
        top_models = get_top_n_models_in_fold(fold_dir, top_n=3)
        if len(top_models) < 3:
            print(f"   ⚠️ Найдено только {len(top_models)} модели. Нужно минимум 3. Пропускаем.")
            continue
            
        val_tfrecord = fold_dir / "data" / "val" / "data.tfrecord"
        val_csv = fold_dir / "data" / "val" / "dataset.csv"
        val_labels = fold_dir / "data" / "val" / "labels.csv"
        
        if not val_tfrecord.exists():
            print(f"   ❌ Не найден TFRecord: {val_tfrecord}")
            continue

        lookback = top_models[0][1]["config"]["lookback"]
        n_features = top_models[0][1]["config"]["features_count"]

        # Создаем датасет
        dataset = tf.data.TFRecordDataset(str(val_tfrecord)).map(
            lambda x: parse_tfrecord_fn(x, lookback, n_features), 
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(4096).prefetch(tf.data.AUTOTUNE)

        # Собираем предсказания каждой модели в словарь
        fold_results = {}

        for i, (model_path, meta, val_loss) in enumerate(top_models, 1):
            print(f"   🔮 Модель {i}: {model_path.name} (Loss: {val_loss:.4f})")
            model = tf.keras.models.load_model(model_path, compile=False)
            preds = model.predict(dataset, verbose=0)
            
            # Сохраняем выходы каждой модели в отдельные колонки
            fold_results[f'm{i}_p0'] = preds[:, 0] # Short
            fold_results[f'm{i}_p1'] = preds[:, 1] # Hold
            fold_results[f'm{i}_p2'] = preds[:, 2] # Long
            
            tf.keras.backend.clear_session()

        df_fold = None
        if val_csv.exists():
            df_prices = pd.read_csv(val_csv)
            # Обязательная сортировка
            df_prices = df_prices.sort_values(by=['ticker', 'datetime']).reset_index(drop=True)
            
            # Воссоздаем логику нарезки окон, чтобы получить точный список индексов
            valid_indices = []
            for ticker, group in df_prices.groupby('ticker'):
                group_len = len(group)
                group_indices = group.index.tolist()
                
                # Точная копия цикла из convert_to_tfrecords.py
                for i in range(group_len - lookback):
                    # Цель берется на индексе i + lookback - 1
                    target_idx = group_indices[i + lookback - 1]
                    valid_indices.append(target_idx)
            
            # Оставляем только те строки, которые реально попали в TFRecord
            df_aligned = df_prices.loc[valid_indices].reset_index(drop=True)
            
            pred_len = len(list(fold_results.values())[0])
            
            if len(df_aligned) == pred_len:
                print("   🔗 Идеальное совпадение! Привязываем вероятности.")
                for col, values in fold_results.items():
                    df_aligned[col] = values
                df_fold = df_aligned
            else:
                print(f"   ❌ ОШИБКА АЛГОРИТМА: Вычислено {len(df_aligned)} строк, а Preds = {pred_len}.")
        else:
            print("   ❌ Не найден dataset.csv")

        if df_fold is not None:
            all_predictions.append(df_fold)

    if all_predictions:
        print("\n🧩 Сборка финального датасета...")
        final_df = pd.concat(all_predictions, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['datetime', 'ticker'], keep='last')
        final_df = final_df.sort_values(by=['ticker', 'datetime']).reset_index(drop=True)
        
        final_df.to_parquet(output_parquet)
        print(f"✅ Готово! Каждая свеча теперь имеет 9 колонок вероятностей (3 модели по 3 класса).")
        print(f"📄 Файл: {output_parquet}")
    else:
        print("❌ Не удалось собрать данные.")

if __name__ == "__main__":
    main()