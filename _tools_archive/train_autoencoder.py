import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from keras.src import Model, Input
from keras.src.layers import Dense, LeakyReLU
# --- ИЗМЕНЕНИЕ №1: Импортируем ReduceLROnPlateau ---
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

# --- Настройка путей ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "2_results"
FEATURES_DIR_PROCESSED = RESULTS_DIR / "features_processed"

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude_cols = {'datetime', 'ticker', 'label', 'meta_target', 'open', 'high', 'low', 'close', 'volume'}
    return [col for col in df.columns if col not in exclude_cols]

def create_autoencoder(input_dim: int, encoding_dim: int, leaky_relu_alpha: float, learning_rate: float) -> tuple[Model, Model]:
    input_layer = Input(shape=(input_dim,), name="input_features")
    encoded = Dense(input_dim // 2, activation='linear')(input_layer)
    encoded = LeakyReLU(negative_slope=leaky_relu_alpha)(encoded)
    encoded = Dense(input_dim // 4, activation='linear')(encoded)
    encoded = LeakyReLU(negative_slope=leaky_relu_alpha)(encoded)
    encoded = Dense(encoding_dim, activation='linear', name="encoded_vector")(encoded)
    decoded = Dense(input_dim // 4, activation='linear')(encoded)
    decoded = LeakyReLU(negative_slope=leaky_relu_alpha)(decoded)
    decoded = Dense(input_dim // 2, activation='linear')(decoded)
    decoded = LeakyReLU(negative_slope=leaky_relu_alpha)(decoded)
    decoded = Dense(input_dim, activation='linear', name="reconstructed_features")(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded, name="Denoising_Autoencoder")
    encoder = Model(inputs=input_layer, outputs=encoded, name="Encoder")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    
    return autoencoder, encoder

def main(args):
    print("--- 🚀 Запуск обучения ШУМОПОДАВЛЯЮЩЕГО автоэнкодера (DAE) ---")
    print("🔥🔥🔥 ВНИМАНИЕ: GPU принудительно отключен, обучение будет идти на CPU. 🔥🔥🔥")
    
    model_dir = RESULTS_DIR / "autoencoders" / args.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    scaler_path = model_dir / "scaler.pkl"
    encoder_path = model_dir / "encoder.keras"
    metadata_path = model_dir / "metadata.json"

    # ... (код загрузки и подготовки данных остается без изменений) ...
    print("Загрузка и объединение файлов с признаками...")
    all_files = list(FEATURES_DIR_PROCESSED.glob("*.csv"))
    df_list = [pd.read_csv(f) for f in tqdm(all_files)]
    full_df = pd.concat(df_list, ignore_index=True)
    full_df['datetime'] = pd.to_datetime(full_df['datetime'])
    full_df.sort_values(by='datetime', inplace=True)
    split_date = full_df['datetime'].max() - pd.DateOffset(years=1)
    train_df = full_df[full_df['datetime'] < split_date]
    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols].values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    print("Обучение MinMaxScaler и масштабирование данных...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Скейлер обучен и сохранен в: {scaler_path}")

    print(f"Добавление Гауссова шума с уровнем (сигма): {args.noise_level}")
    noise = np.random.normal(loc=0.0, scale=args.noise_level, size=X_train_scaled.shape)
    X_train_noisy = X_train_scaled + noise
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)

    print("Создание архитектуры DAE...")
    autoencoder, encoder = create_autoencoder(
        input_dim=len(feature_cols),
        encoding_dim=args.encoding_dim,
        leaky_relu_alpha=args.leaky_relu_alpha,
        learning_rate=args.lr
    )
    autoencoder.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    # --- ИЗМЕНЕНИЕ №2: Создаем callback для понижения скорости обучения ---
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2,      # Уменьшаем LR в 5 раз
        patience=5,      # Ждем 5 эпох без улучшений
        verbose=1,       # Сообщаем об изменениях
        min_lr=1e-6      # Минимальная скорость обучения
    )
    
    print("Начало обучения DAE (шум -> чистые)...")
    history = autoencoder.fit(
        X_train_noisy,
        X_train_scaled,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        validation_split=0.1,
        # --- ИЗМЕНЕНИЕ №3: Добавляем новый callback в список ---
        callbacks=[early_stopping, reduce_lr_callback],
        verbose=1
    )

    min_val_loss = min(history.history.get('val_loss', [np.inf]))
    print(f"✅ Обучение завершено. Минимальный val_loss: {min_val_loss:.6f}")
    
    encoder.save(encoder_path)
    print(f"✅ Модель энкодера сохранена в: {encoder_path}")
    
    metadata = {
        "model_name": args.model_name,
        "type": "DenoisingAutoencoder",
        "noise_level": args.noise_level,
        "source_features_count": len(feature_cols),
        "encoded_dim": args.encoding_dim,
        "min_validation_loss": min_val_loss,
        "source_feature_names": feature_cols
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Метаданные сохранены в: {metadata_path}")
    print("\n🎉 Пайплайн обучения DAE успешно завершен!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Denoising Autoencoder for feature extraction.")
    parser.add_argument("--model_name", type=str, required=True, help="A unique name for this autoencoder model.")
    parser.add_argument("--encoding_dim", type=int, default=30, help="The dimensionality of the encoded representation.")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size for training.")
    parser.add_argument("--leaky_relu_alpha", type=float, default=0.2, help="Alpha parameter for LeakyReLU activation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--noise_level", type=float, default=0.1, help="Standard deviation of Gaussian noise to add to the input data.")
    
    args = parser.parse_args()
    main(args)