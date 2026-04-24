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
    sequence = tf.io.parse_tensor(parsed['sequence'], out_type=tf.float32)
    sequence.set_shape([lookback, n_features])
    return sequence

def get_top_n_models_in_fold(fold_dir, top_n=3):
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
        except Exception: pass
            
    valid_models.sort(key=lambda x: x[2])
    return valid_models[:top_n]

def main():
    # Настройки путей
    base_dataset_dir = Path("data/processed/2000_2026_1d_60_10")
    output_dir = Path("data/processed/2000_2026_1d/rl_env")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_parquet = output_dir / "environment_data.parquet"

    print(f"🚀 Multi-Model Inference (Top-3) -> {base_dataset_dir.name}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

    folds = sorted([d for d in base_dataset_dir.glob("fold_*") if d.is_dir()])
    all_predictions = []

    for fold_dir in folds:
        print(f"\n📂 Фолд: {fold_dir.name}")
        
        top_models = get_top_n_models_in_fold(fold_dir, top_n=3)
        if len(top_models) < 3: continue
            
        val_tfrecord = fold_dir / "data" / "val" / "data.tfrecord"
        val_data_path = fold_dir / "data" / "val" / "ml_data.parquet"
        
        if not val_tfrecord.exists() or not val_data_path.exists():
            print(f"   ⚠️ Пропуск: нет файлов в {fold_dir}")
            continue

        lookback = top_models[0][1]["config"]["lookback"]
        n_features = top_models[0][1]["config"]["features_count"]

        # 1. Инференс
        dataset = tf.data.TFRecordDataset(str(val_tfrecord)).map(
            lambda x: parse_tfrecord_fn(x, lookback, n_features), 
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(4096).prefetch(tf.data.AUTOTUNE)

        fold_results = {}
        for i, (model_path, meta, val_loss) in enumerate(top_models, 1):
            print(f"   🔮 Model {i} (Loss: {val_loss:.4f})")
            model = tf.keras.models.load_model(model_path, compile=False)
            preds = model.predict(dataset, verbose=0)
            fold_results[f'm{i}_p0'], fold_results[f'm{i}_p1'], fold_results[f'm{i}_p2'] = preds[:, 0], preds[:, 1], preds[:, 2]
            tf.keras.backend.clear_session()

        # 2. Математически точная привязка (синхронно с convert_to_tfrecords.py)
        df_src = pd.read_parquet(val_data_path)
        df_src = df_src.sort_values(by=['ticker', 'datetime']).reset_index(drop=True)
        
        valid_indices = []
        for ticker, group in df_src.groupby('ticker'):
            idx_list = group.index.tolist()
            # Повторяем логику: i от 0 до len-lookback. Цель на i+lookback-1
            for i in range(len(idx_list) - lookback):
                valid_indices.append(idx_list[i + lookback - 1])
        
        df_aligned = df_src.loc[valid_indices].reset_index(drop=True)
        
        # Проверка финальной длины
        expected_len = len(list(fold_results.values())[0])
        if len(df_aligned) == expected_len:
            print(f"   ✅ Выравнивание успешно ({expected_len} строк)")
            for col, values in fold_results.items():
                df_aligned[col] = values
            all_predictions.append(df_aligned)
        else:
            print(f"   ❌ Ошибка: Вычислено {len(df_aligned)}, получено {expected_len}")

    if all_predictions:
        final_df = pd.concat(all_predictions, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['datetime', 'ticker'], keep='last')
        final_df.sort_values(by=['ticker', 'datetime']).to_parquet(output_parquet)
        print(f"\n🏁 Итог: {output_parquet} ({len(final_df)} строк)")

if __name__ == "__main__":
    main()