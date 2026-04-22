import argparse
import json
import sys
import os
import time
from pathlib import Path
import warnings
import numpy as np

# Прячем системный спам
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Скрываем INFO и WARNING от C++ ядра TF
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')         # Скрываем Python-ворнинги

import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed precision включена!")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, GRU, Bidirectional, Dense, Dropout, Attention,
    Add, LayerNormalization, GlobalAveragePooling1D, LeakyReLU
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras import regularizers

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Динамическое выделение видеопамяти включено!")
    except RuntimeError as e:
        print(e)

class SmartBacktrackCallback(Callback):
    def __init__(self, best_weights_path, monitor_loss='val_loss', factor=0.5, patience=4, min_lr=1e-6, max_rollbacks=2, stop_threshold=1.1):
        super(SmartBacktrackCallback, self).__init__()
        self.monitor_loss = monitor_loss
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.max_rollbacks = max_rollbacks
        self.best_weights_path = str(best_weights_path)
        self.stop_threshold = stop_threshold
        
        self.wait = 0
        self.rollback_count = 0
        self.best_loss = np.inf
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor_loss)
        if current_loss is None: return

        if epoch > 10 and current_loss > self.stop_threshold and self.wait >= self.patience:
            print(f"\n⚠️ Итерация безнадежна (val_loss {current_loss:.4f} > {self.stop_threshold}). Пропускаем.")
            self.model.stop_training = True
            return

        if current_loss < self.best_loss - 1e-4:
            self.best_loss = current_loss
            self.best_epoch = epoch + 1
            self.wait = 0
            self.rollback_count = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.rollback_count += 1
                
                if self.rollback_count > self.max_rollbacks:
                    print(f"\n🛑 Итерация застряла на плато. Досрочное завершение.")
                    self.model.stop_training = True
                    return

                if os.path.exists(self.best_weights_path):
                    print(f"\n⚠️ Откат к весам из эпохи #{self.best_epoch} (Откат {self.rollback_count}/{self.max_rollbacks})...")
                    self.model.load_weights(self.best_weights_path)

                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                if old_lr > self.min_lr:
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    self.model.optimizer.learning_rate.assign(new_lr)
                    print(f"📉 Снижаю learning rate до {new_lr:.0e}.")
                    self.wait = 0

def parse_tfrecord_fn(example, lookback, n_features):
    feature_description = {
        'sequence': tf.io.FixedLenFeature([], tf.string), 
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    sequence = tf.io.parse_tensor(example['sequence'], out_type=tf.float32)
    sequence.set_shape([lookback, n_features])
    label = tf.one_hot(example['target'], depth=3)
    label.set_shape([3])
    return sequence, label

def load_tfrecord_dataset(file_path, batch_size, lookback, n_features, is_training=True):
    dataset = tf.data.TFRecordDataset(str(file_path), num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: parse_tfrecord_fn(x, lookback, n_features), num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.cache().shuffle(10000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def compute_class_weights_fast(tfrecord_path):
    print("⚙️ Расчет идеальных весов классов...")
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    class_counts = {0: 0, 1: 0, 2: 0}
    feature_description = {'target': tf.io.FixedLenFeature([], tf.int64)}
    
    for raw_record in dataset:
        parsed = tf.io.parse_single_example(raw_record, feature_description)
        class_counts[int(parsed['target'].numpy())] += 1
        
    total = sum(class_counts.values())
    weights = {c: total / (3.0 * max(1, count)) for c, count in class_counts.items()}
    print(f"   Баланс: SL(0)={class_counts[0]}, Hold(1)={class_counts[1]}, TP(2)={class_counts[2]}")
    print(f"   Веса:   SL(0)={weights[0]:.2f}, Hold(1)={weights[1]:.2f}, TP(2)={weights[2]:.2f}")
    return weights

def create_model(seq_len, n_features, l2_reg=1e-5):
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    x = Dense(32, activation='linear', name='fp16_projection')(inputs)
    x = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg)))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg)))(x)
    res_x = Dropout(0.2)(x)
    attn_out = Attention()([res_x, res_x])
    x = Add()([res_x, attn_out])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(3, activation='softmax', name='out', dtype='float32')(x)
    return Model(inputs=inputs, outputs=outputs)

def save_record_model(model, history, acc, loss, train_time, run, args, seq_len, models_dir):
    # Добавляем timestamp (время), чтобы файлы при --append не перезаписывали друг друга при совпадении номера run
    timestamp = int(time.time())
    model_filename = f"trading_bot_best_acc_{acc*100:.2f}_run{run}_{timestamp}.keras"
    model_filepath = models_dir / model_filename
    model.save(model_filepath)
    
    meta_filepath = models_dir / f"trading_bot_best_acc_{acc*100:.2f}_run{run}_{timestamp}_meta.json"
    clean_history = {k: [float(val) for val in v] for k, v in history.history.items()} if history else {}
    
    report = {
        "fold": args.fold,
        "run_id": str(run),
        "timestamp": timestamp,
        "best_val_accuracy": float(acc),
        "val_loss": float(loss),
        "training_time_seconds": float(train_time),
        "hyperparameters": {
            "lookback_window": seq_len,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "start_lr": args.lr,
            "l2_reg": args.l2_reg
        },
        "training_history": clean_history
    }
    
    with open(meta_filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    print(f"🏆 НОВЫЙ РЕКОРД СОХРАНЕН: {model_filename}")

def main(args):
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASET_DIR = BASE_DIR / args.dataset_dir
    FOLD_DIR = DATASET_DIR / args.fold
    
    TFRECORDS_DIR = FOLD_DIR / "data"
    MODELS_DIR = FOLD_DIR / "models"
    ARTIFACTS_DIR = FOLD_DIR / "artifacts"
    
    # --- ЛОГИКА ПРОВЕРКИ СУЩЕСТВУЮЩИХ МОДЕЛЕЙ ---
    if MODELS_DIR.exists() and any(MODELS_DIR.glob("*.keras")):
        if args.force:
            print(f"⚠️ Флаг --force: Очищаем папку моделей фолда [{args.fold}]...")
            for f in MODELS_DIR.glob("*"): 
                f.unlink()
        elif args.append:
            print(f"➕ Флаг --append: Модели в фолде [{args.fold}] существуют. Обучаем новые поверх старых.")
        else:
            print(f"✅ В фолде [{args.fold}] уже есть обученные модели.")
            print("⏭️ Пропуск обучения. Используйте --force (перезапись) или --append (добавление).")
            return
                
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Старт обучения. Фолд: [{args.fold}]")
    
    dataset_meta_path = DATASET_DIR / "metadata.json"
    if not dataset_meta_path.exists():
        print(f"❌ Ошибка: {dataset_meta_path} не найден.")
        return
    with open(dataset_meta_path, 'r', encoding='utf-8') as f:
        seq_len = json.load(f)["parameters"]["lookback"]

    features_json = ARTIFACTS_DIR / "features_selected.json"
    if not features_json.exists():
        print(f"❌ Ошибка: {features_json} не найден.")
        return
    with open(features_json, 'r', encoding='utf-8') as f:
        n_features = len(json.load(f).get("feature_order", []))
            
    print(f"📊 Форма данных: [Lookback: {seq_len}, Features: {n_features}]")
    
    train_record_path = TFRECORDS_DIR / "train" / "data.tfrecord"
    val_record_path = TFRECORDS_DIR / "val" / "data.tfrecord"
    
    class_weights_dict = compute_class_weights_fast(train_record_path)
    
    print("⏳ Подготовка конвейера данных...")
    train_dataset = load_tfrecord_dataset(train_record_path, args.batch_size, seq_len, n_features, is_training=True)
    val_dataset = load_tfrecord_dataset(val_record_path, args.batch_size, seq_len, n_features, is_training=False)

    global_best_acc = 0.0

    for run in range(1, args.runs + 1):
        print(f"\n{'-'*50}\n🔄 ИТЕРАЦИЯ {run}/{args.runs} (Лучшая точность сессии: {global_best_acc*100:.2f}%)\n{'-'*50}")
        
        tf.keras.backend.clear_session()
        model = create_model(seq_len, n_features, args.l2_reg)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        temp_weights_path = MODELS_DIR / f"temp_best_run_{int(time.time())}.weights.h5"
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, verbose=0, restore_best_weights=True),
            ModelCheckpoint(filepath=temp_weights_path, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True, verbose=0),
            SmartBacktrackCallback(best_weights_path=temp_weights_path, monitor_loss='val_loss', patience=4, factor=0.5, min_lr=1e-6, max_rollbacks=2)
        ]

        start_time = time.time()
        try:
            history = model.fit(
                train_dataset, 
                epochs=args.epochs, 
                validation_data=val_dataset, 
                callbacks=callbacks, 
                class_weight=class_weights_dict, 
                verbose=2
            )
        except KeyboardInterrupt:
            print("\n⚠️ Multi-Run прерван пользователем.")
            break

        train_time = time.time() - start_time

        if os.path.exists(temp_weights_path):
            model.load_weights(temp_weights_path)
            os.remove(temp_weights_path)
            
        loss, acc = model.evaluate(val_dataset, verbose=0)
        print(f"\n🎯 Точность итерации {run}: {acc*100:.2f}%")

        if acc > global_best_acc:
            global_best_acc = acc
            save_record_model(model, history, acc, loss, train_time, run, args, seq_len, MODELS_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/processed/2000_2026_1d", help="Папка датасета")
    parser.add_argument("--fold", type=str, default="fold_2010", help="Имя фолда для обучения")
    parser.add_argument("--runs", type=int, default=10, help="Количество циклов инициализации весов")
    parser.add_argument("--batch_size", type=int, default=512, help="Размер батча")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох")
    parser.add_argument("--lr", type=float, default=1e-3, help="Стартовый Learning Rate")
    parser.add_argument("--l2_reg", type=float, default=1e-5, help="L2 регуляризация")
    parser.add_argument("--force", action="store_true", help="Принудительное обучение с удалением старых моделей")
    parser.add_argument("--append", action="store_true", help="Добавить новые модели, не удаляя старые")
    args = parser.parse_args()
    main(args)