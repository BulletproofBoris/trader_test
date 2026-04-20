# _tools/measure_sharpness.py

import argparse
import numpy as np
import os
from pathlib import Path
import warnings
import json
import pickle
from tqdm import tqdm # <--- ИСПРАВЛЕНО: Добавлен недостающий импорт

# --- Подавление лишних сообщений ---
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# --- Автономное определение путей ---
try:
    PROJECT_DIR = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_DIR = Path.cwd()

RESULTS_DIR = PROJECT_DIR / "2_results"
MODELS_DIR = RESULTS_DIR / "models"
SCALERS_DIR = RESULTS_DIR / "scalers"
TFRECORDS_DIR = RESULTS_DIR
# --- Конец блока ---

def parse_tfrecord_fn(example):
    """Парсит один пример из TFRecord файла."""
    feature_description = { 'sequence': tf.io.FixedLenFeature([], tf.string), 'target': tf.io.FixedLenFeature([], tf.int64) }
    example = tf.io.parse_single_example(example, feature_description)
    sequence = tf.io.parse_tensor(example['sequence'], out_type=tf.float32)
    label = example['target']
    return sequence, label

def load_tfrecord_dataset(file_path, batch_size):
    """Загружает и подготавливает датасет из TFRecord файла."""
    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # Важно: НЕ используем .repeat() для оценки, нам нужен один проход по всем данным
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main(args):
    model_name = args.model_name
    model_path = MODELS_DIR / model_name / "model.keras" 
    scaler_path = SCALERS_DIR / f"{model_name}_scaler.pkl"
    tfrecord_base_path = TFRECORDS_DIR / f"tfrecord_{model_name}"
    meta_path = tfrecord_base_path / "metadata.json"

    print(f"--- 📏 Детерминированное измерение резкости для модели: '{model_name}' ---")
    try:
        model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
        with open(meta_path, 'r') as f: metadata = json.load(f)
        print("✅ Модель, скейлер и метаданные успешно загружены.")
    except Exception as e:
        print(f"❌ Ошибка при загрузке артефактов: {e}"); return
        
    seq_len, n_features = metadata['lookback_period'], metadata['num_features']

    # --- Пайплайн подготовки данных ---
    val_file = tfrecord_base_path / "test.tfrecord"
    val_dataset = load_tfrecord_dataset(str(val_file), args.batch_size)
    
    @tf.function
    def scale_features(features, label):
        original_shape = tf.shape(features)
        reshaped_features = tf.reshape(features, [-1, n_features])
        scaled_reshaped = tf.py_function(func=scaler.transform, inp=[reshaped_features], Tout=tf.float32)
        scaled_features = tf.reshape(scaled_reshaped, original_shape)
        scaled_features.set_shape([None, seq_len, n_features]); label.set_shape([None])
        return scaled_features, label

    val_dataset_scaled = val_dataset.map(scale_features, num_parallel_calls=tf.data.AUTOTUNE)
    print("✅ Тестовый датасет подготовлен.")

    # 1. Считаем базовый loss на всем тестовом наборе
    print("\nРасчет базовой ошибки на всем тестовом наборе...")
    loss_base, acc_base = model.evaluate(val_dataset_scaled, verbose=0)
    print(f"  - Базовый Loss: {loss_base:.4f}")

    # 2. Несколько раз "толкаем" модель и усредняем результат
    perturbed_losses = []
    center_weights = model.get_weights()
    
    print(f"\nВыполнение {args.n_probes} тестовых 'толчков' модели...")
    for i in tqdm(range(args.n_probes), desc="Пробы"):
        # Создаем случайный шум, нормированный как веса
        noise = [np.random.normal(0, 1, w.shape) for w in center_weights]
        for n, w in zip(noise, center_weights):
            n_norm = np.linalg.norm(n.flatten())
            w_norm = np.linalg.norm(w.flatten())
            if n_norm > 0:
                n *= (w_norm / n_norm) * args.noise_scale
        
        perturbed_weights = [w + n for w, n in zip(center_weights, noise)]
        model.set_weights(perturbed_weights)
        
        loss_perturbed, _ = model.evaluate(val_dataset_scaled, verbose=0)
        perturbed_losses.append(loss_perturbed)

    avg_perturbed_loss = np.mean(perturbed_losses)
    print(f"  - Средний Loss после 'толчков': {avg_perturbed_loss:.4f}")

    # 3. Считаем итоговый индекс резкости
    sharpness_index = avg_perturbed_loss - loss_base

    print("\n--- 📊 Итоги ---")
    print(f"Индекс резкости (чем ниже, тем лучше): {sharpness_index:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Детерминированное измерение резкости (устойчивости) минимума модели.")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--n_probes', type=int, default=10, help="Количество случайных 'толчков' для усреднения результата.")
    parser.add_argument('--noise_scale', type=float, default=1e-3, help="Масштаб шума для 'толчка' (относительно нормы весов).")
    args = parser.parse_args()
    main(args)