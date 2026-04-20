import argparse
import json
import joblib
from pathlib import Path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
from tqdm import tqdm
import tensorflow as tf

# --- Настройка путей ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "2_results"
FEATURES_DIR_PROCESSED = RESULTS_DIR / "features_processed"

def main(args):
    print("--- 👽 Запуск извлечения 'супер-признаков' с помощью энкодера ---")
    print("🔥🔥🔥 ВНИМАНИЕ: GPU принудительно отключен, извлечение будет идти на CPU. 🔥🔥🔥")

    autoencoder_dir = RESULTS_DIR / "autoencoders" / args.model_name
    encoder_path = autoencoder_dir / "encoder.keras"
    scaler_path = autoencoder_dir / "scaler.pkl"
    metadata_path = autoencoder_dir / "metadata.json"
    output_dir = RESULTS_DIR / "features_encoded" / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not all([encoder_path.exists(), scaler_path.exists(), metadata_path.exists()]):
        print(f"❌ Ошибка: Не найдены артефакты в папке {autoencoder_dir}.")
        return

    print("Загрузка энкодера, скейлера и метаданных...")
    encoder = tf.keras.models.load_model(encoder_path)
    scaler = joblib.load(scaler_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Это наш "канонический" список из 132 признаков
    source_feature_names = metadata["source_feature_names"]
    encoded_dim = metadata["encoded_dim"]
    encoded_feature_names = [f"AE_{i}" for i in range(encoded_dim)]
    
    print(f"✅ Артефакты для модели '{args.model_name}' успешно загружены.")

    all_files = list(FEATURES_DIR_PROCESSED.glob("*.csv"))
    print(f"Найдено {len(all_files)} файлов для обработки из {FEATURES_DIR_PROCESSED}")

    possible_meta_cols = ['datetime', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'label', 'meta_target']

    for file_path in tqdm(all_files, desc="Преобразование признаков"):
        df = pd.read_csv(file_path)
        
        existing_meta_cols = [col for col in possible_meta_cols if col in df.columns]
        df_meta = df[existing_meta_cols].copy()
        
        # --- 👇👇👇 ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ 👇👇👇 ---
        # 1. Берем только те признаки, которые есть в каноническом списке
        existing_features = [col for col in source_feature_names if col in df.columns]
        X = df[existing_features].copy()

        # 2. Выравнивание данных: добавляем недостающие колонки и заполняем их нулями
        missing_features = set(source_feature_names) - set(existing_features)
        if missing_features:
            for feature in missing_features:
                X[feature] = 0
        
        # 3. Гарантируем правильный порядок колонок
        X = X[source_feature_names]
        # --- 👆👆👆 КОНЕЦ ИСПРАВЛЕНИЯ 👆👆👆 ---
        
        X_scaled = scaler.transform(X.values)
        X_encoded = encoder.predict(X_scaled, verbose=0)
        
        df_encoded_features = pd.DataFrame(X_encoded, columns=encoded_feature_names, index=df.index)
        final_df = pd.concat([df_meta, df_encoded_features], axis=1)
        
        output_path = output_dir / file_path.name
        final_df.to_csv(output_path, index=False)

    print(f"\n✅ Процесс завершен. {len(all_files)} файлов с 'супер-признаками' сохранено в: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features using a trained autoencoder.")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="The name of the trained autoencoder model to use.")
    args = parser.parse_args()
    main(args)