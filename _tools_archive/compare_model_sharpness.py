# _tools/compare_model_sharpness.py
import argparse
import numpy as np
import pandas as pd
import os
from pathlib import Path
import warnings
import json
from tqdm import tqdm

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
TFRECORDS_DIR = RESULTS_DIR
# --- Конец блока ---

# --- ИСПРАВЛЕНО: Функции загрузки данных теперь идентичны скриптам обучения ---
def parse_tfrecord_fn(example, lookback, n_features):
    """Парсит один пример из TFRecord файла, зная его структуру."""
    feature_description = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    sequence = tf.io.parse_tensor(example['sequence'], out_type=tf.float32)
    label = example['target']
    
    sequence.set_shape([lookback, n_features])
    # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Преобразуем метку в one-hot формат ---
    label = tf.one_hot(label, depth=3)
    label.set_shape([3])
    return sequence, label

def load_tfrecord_dataset(file_path, batch_size, lookback, n_features):
    """Загружает и подготавливает датасет из TFRecord файла."""
    dataset = tf.data.TFRecordDataset(str(file_path))
    parser = lambda x: parse_tfrecord_fn(x, lookback, n_features)
    dataset = dataset.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
# --- КОНЕЦ БЛОКА ИСПРАВЛЕНИЙ ---


def measure_sharpness_for_model(model_name: str, data_model_name: str, args: argparse.Namespace):
    """Загружает одну модель и детерминированно измеряет ее loss, accuracy и sharpness."""
    model_path = MODELS_DIR / model_name / "model.keras"
    tfrecord_base_path = TFRECORDS_DIR / f"tfrecord_{data_model_name}"
    meta_path = tfrecord_base_path / "metadata.json"

    try:
        model = tf.keras.models.load_model(model_path)
        with open(meta_path, 'r') as f: metadata = json.load(f)
    except Exception as e:
        print(f"  [WARN] Пропуск модели {model_name}: не удалось загрузить артефакты. Ошибка: {e}")
        return None
        
    try:
        seq_len, n_features = metadata['lookback'], metadata['n_features']
    except KeyError:
        print(f"  [WARN] Пропуск модели {model_name}: в metadata.json отсутствуют ключи 'lookback' или 'n_features'.")
        return None

    val_file = tfrecord_base_path / "test.tfrecord"
    # --- ИСПРАВЛЕНО: Передаем seq_len и n_features для корректной загрузки ---
    val_dataset = load_tfrecord_dataset(str(val_file), args.batch_size, seq_len, n_features)
    
    # --- ИСПРАВЛЕНО: Удалена лишняя логика масштабирования. Данные в TFRecord уже отмасштабированы. ---
    
    # --- Оценка базовой производительности ---
    loss_base, acc_base = model.evaluate(val_dataset, verbose=0)
    
    perturbed_losses = []
    center_weights = model.get_weights()
    
    # --- Расчет "резкости" ---
    for _ in range(args.n_probes):
        noise = [np.random.normal(0, 1, w.shape) for w in center_weights]
        for n, w in zip(noise, center_weights):
            n_norm, w_norm = np.linalg.norm(n.flatten()), np.linalg.norm(w.flatten())
            if n_norm > 0 and w_norm > 0:
                n *= (w_norm / n_norm) * args.noise_scale
        perturbed_weights = [w + n for w, n in zip(center_weights, noise)]
        model.set_weights(perturbed_weights)
        loss_perturbed, _ = model.evaluate(val_dataset, verbose=0)
        perturbed_losses.append(loss_perturbed)

    sharpness_index = np.mean(perturbed_losses) - loss_base
    
    return {"model_name": model_name, "min_loss": loss_base, "accuracy": acc_base, "sharpness": sharpness_index}

def main(args):
    model_prefix = args.model_prefix
    print(f"--- Сравнительный детерминированный анализ резкости для моделей с префиксом: '{model_prefix}' ---")
    
    model_dirs = sorted([d for d in MODELS_DIR.glob(f"{model_prefix}*") if d.is_dir()])
    if not model_dirs:
        print(f"❌ Не найдено папок моделей для префикса '{model_prefix}' в {MODELS_DIR}"); return
        
    print(f"Найдено {len(model_dirs)} моделей для сравнения.")
    
    if args.data_model_name:
        print(f"Используются данные от: '{args.data_model_name}' (для всех моделей)")
    else:
        print("Используются данные, соответствующие каждой модели.")

    all_results = []
    for model_dir in tqdm(model_dirs, desc="Анализ моделей"):
        data_source_name = args.data_model_name if args.data_model_name else model_dir.name
        
        metrics = measure_sharpness_for_model(model_dir.name, data_source_name, args)
        if metrics:
            all_results.append(metrics)

    if not all_results:
        print("\nНе удалось собрать данные для анализа."); return
        
    results_df = pd.DataFrame(all_results).set_index('model_name')
    
    print("\n--- 📊 Итоговая таблица сравнения (детерминированная) ---")
    print(results_df.to_string(float_format="%.4f", columns=["min_loss", "accuracy", "sharpness"]))
    
    best_loss_model = results_df['min_loss'].idxmin()
    best_sharpness_model = results_df['sharpness'].idxmin()
    best_accuracy_model = results_df['accuracy'].idxmax()
    
    print("\n--- 🏆 Рекомендации ---")
    print(f"Модель с самым низким loss (лучшее качество): {best_loss_model} ({results_df.loc[best_loss_model, 'min_loss']:.4f})")
    print(f"Модель с самой высокой точностью (best accuracy): {best_accuracy_model} ({results_df.loc[best_accuracy_model, 'accuracy']:.4f})")
    print(f"Модель с наименьшей резкостью (самая устойчивая): {best_sharpness_model} ({results_df.loc[best_sharpness_model, 'sharpness']:.4f})")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сравнительный детерминированный анализ резкости моделей.")
    parser.add_argument('--model_prefix', type=str, required=True)
    parser.add_argument('--data_model_name', type=str, default=None, help="Имя модели, чьи данные (TFRecord) использовать для анализа. Если не указано, каждая модель использует свои данные.")
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--n_probes', type=int, default=10)
    parser.add_argument('--noise_scale', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)