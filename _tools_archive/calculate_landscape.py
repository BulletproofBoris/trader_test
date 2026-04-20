# _tools/calculate_landscape.py

import argparse
import json
import pickle
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import warnings
import pandas as pd
import itertools

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
LANDSCAPES_DIR = RESULTS_DIR / "landscape_data"
# --- Конец блока ---

def parse_tfrecord_fn(example):
    feature_description = { 'sequence': tf.io.FixedLenFeature([], tf.string), 'target': tf.io.FixedLenFeature([], tf.int64) }
    example = tf.io.parse_single_example(example, feature_description)
    sequence = tf.io.parse_tensor(example['sequence'], out_type=tf.float32)
    label = example['target']
    return sequence, label

def load_tfrecord_dataset(file_path, batch_size):
    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_random_directions(weights, n_slices):
    directions = []
    for _ in range(n_slices):
        dir1 = [np.random.randn(*w.shape) for w in weights]
        dir2 = [np.random.randn(*w.shape) for w in weights]
        for d1, d2, w in zip(dir1, dir2, weights):
            d1_flat, d2_flat = d1.flatten(), d2.flatten()
            w_norm = np.linalg.norm(w.flatten())
            if np.linalg.norm(d1_flat) > 0: d1_flat *= w_norm / np.linalg.norm(d1_flat)
            if np.dot(d1_flat, d1_flat) > 0: d2_flat -= np.dot(d1_flat, d2_flat) / np.dot(d1_flat, d1_flat) * d1_flat
            if np.linalg.norm(d2_flat) > 0: d2_flat *= w_norm / np.linalg.norm(d2_flat)
            np.copyto(d1, d1_flat.reshape(d1.shape)); np.copyto(d2, d2_flat.reshape(d2.shape))
        directions.append((dir1, dir2))
    return directions

def perturb_weights(center_weights, dir1, dir2, x, y):
    return [w + d1 * x + d2 * y for w, d1, d2 in zip(center_weights, dir1, dir2)]

def main(args):
    model_name_to_analyze = args.model_name
    output_subdir = LANDSCAPES_DIR / model_name_to_analyze
    output_subdir.mkdir(parents=True, exist_ok=True)

    # --- ИЗМЕНЕНО: Гибкое определение имени модели для ДАННЫХ ---
    data_source_model_name = args.data_model_name if args.data_model_name else model_name_to_analyze
    
    model_path = MODELS_DIR / model_name_to_analyze / args.model_filename
    scaler_path = SCALERS_DIR / f"{data_source_model_name}_scaler.pkl"
    tfrecord_base_path = RESULTS_DIR / f"tfrecord_{data_source_model_name}"
    meta_path = tfrecord_base_path / "metadata.json"

    print(f"--- Расчет ландшафта потерь для модели: {model_path.name} ---")
    print(f"--- Используются данные от: {data_source_model_name} ---")
    try:
        model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
        with open(meta_path, 'r') as f: metadata = json.load(f)
        print("✅ Модель, скейлер и метаданные успешно загружены.")
    except Exception as e:
        print(f"❌ Ошибка при загрузке артефактов: {e}"); return
        
    seq_len, n_features = metadata['lookback_period'], metadata['num_features']
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
    x_batch, y_batch = next(iter(val_dataset_scaled))
    print("✅ Данные для оценки подготовлены.")

    center_weights = model.get_weights()
    direction_vectors = create_random_directions(center_weights, args.num_images)

    for i, (dir1, dir2) in enumerate(direction_vectors):
        slice_num = i + 1
        print(f"\n--- Расчет данных для среза №{slice_num}/{args.num_images} ---")
        results = []
        
        grid_coords = np.linspace(-args.scale, args.scale, args.points)
        grid_points_product = list(itertools.product(grid_coords, grid_coords))
        
        for x_coord, y_coord in tqdm(grid_points_product, desc=f"Расчет среза {slice_num}"):
            perturbed_weights = perturb_weights(center_weights, dir1, dir2, x_coord, y_coord)
            model.set_weights(perturbed_weights)
            loss, acc = model.evaluate(x=x_batch, y=y_batch, verbose=0)
            results.append({'x': x_coord, 'y': y_coord, 'loss': loss, 'accuracy': acc})
        
        results_df = pd.DataFrame(results)
        model_basename = Path(args.model_filename).stem
        output_filename = output_subdir / f"landscape_data_{model_basename}_run_{slice_num}.npz"
        
        np.savez_compressed(output_filename, x=results_df['x'].to_numpy(), y=results_df['y'].to_numpy(),
                            loss=results_df['loss'].to_numpy(), accuracy=results_df['accuracy'].to_numpy())
        print(f"✅ Результаты среза сохранены в: {output_filename}")

    print("\n✅ Расчет ландшафта полностью завершен.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Расчет ландшафта потерь для обученной модели.")
    parser.add_argument('--model_name', type=str, required=True, help="Имя модели для АНАЛИЗА (папка).")
    # --- ДОБАВЛЕН НОВЫЙ АРГУМЕНТ ---
    parser.add_argument('--data_model_name', type=str, default=None, help="Имя модели, чьи ДАННЫЕ использовать.")
    parser.add_argument('--model_filename', type=str, default="model.keras", help="Имя файла модели для анализа.")
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--num_images', type=int, default=1)
    parser.add_argument('--points', type=int, default=10)
    parser.add_argument('--scale', type=float, default=1.0)
    args = parser.parse_args()
    main(args)