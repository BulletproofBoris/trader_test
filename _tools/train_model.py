import argparse
import json
import sys
import os
import time
from pathlib import Path
import warnings
import numpy as np

# Прячем логи TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
    def __init__(self, best_weights_path, monitor_loss='val_loss', factor=0.5, patience=4, min_lr=1e-6, max_rollbacks=2):
        super(SmartBacktrackCallback, self).__init__()
        self.monitor_loss = monitor_loss
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.max_rollbacks = max_rollbacks
        self.best_weights_path = str(best_weights_path)
        
        self.wait = 0
        self.rollback_count = 0
        self.best_loss = np.inf
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor_loss)
        if current_loss is None: return

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
                    print(f'   📉 Снижаю learning rate до {new_lr:.0e}.')
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

def load_tfrecord_dataset(file_path, batch_size, lookback, n_features):
    dataset = tf.data.TFRecordDataset(str(file_path), num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: parse_tfrecord_fn(x, lookback, n_features), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

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
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(3, activation='softmax', name='out', dtype='float32')(x)
    return Model(inputs=inputs, outputs=outputs)

def save_record_model(model, history, acc, loss, train_time, run, args, seq_len, models_dir, exp_config):
    model_filename = f"trading_bot_best_acc_{acc*100:.2f}_run{run}.keras"
    model_filepath = models_dir / model_filename
    model.save(model_filepath)
    
    meta_filepath = models_dir / f"trading_bot_best_acc_{acc*100:.2f}_run{run}_meta.json"
    clean_history = {k: [float(val) for val in v] for k, v in history.history.items()} if history else {}
    
    report = {
        "experiment_info": exp_config,
        "run_id": str(run),
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
        
    print(f"🏆 НОВЫЙ ГЛОБАЛЬНЫЙ РЕКОРД! Модель сохранена: {model_filename}")

def main(args):
    BASE_DIR = Path(__file__).resolve().parent.parent
    EXP_DIR = BASE_DIR / "experiments" / args.exp_name
    TFRECORDS_DIR = EXP_DIR / "tfrecords"
    MODELS_DIR = EXP_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    config_path = EXP_DIR / "exp_config.json"
    if not config_path.exists():
        print(f"❌ Паспорт эксперимента не найден: {config_path}"); return
        
    with open(config_path, 'r', encoding='utf-8') as f:
        exp_config = json.load(f)
            
    print(f"🚀 Старт обучения. Эксперимент: [{args.exp_name}]")
    
    with open(TFRECORDS_DIR / "metadata.json", 'r') as f:
        metadata = json.load(f)
        
    seq_len = metadata['lookback']
    n_features = metadata['n_features']
    class_weights_dict = {int(k): v for k, v in metadata['class_weights'].items()}
    
    print("⏳ Загрузка датасетов в память GPU...")
    train_dataset = load_tfrecord_dataset(TFRECORDS_DIR / "train.tfrecord", args.batch_size, seq_len, n_features)
    val_dataset = tf.data.TFRecordDataset(str(TFRECORDS_DIR / "test.tfrecord"))
    val_dataset = val_dataset.map(lambda x: parse_tfrecord_fn(x, seq_len, n_features), num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    global_best_acc = 0.0
    global_best_run = "None"

    for run in range(1, args.runs + 1):
        print(f"\n{'='*60}\n🔄 ИТЕРАЦИЯ {run} / {args.runs} (Рекорд: {global_best_acc*100:.2f}%)\n{'='*60}")
        
        tf.keras.backend.clear_session()
        model = create_model(seq_len, n_features, args.l2_reg)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        if run == 1: model.summary()
        
        temp_weights_path = MODELS_DIR / f"temp_best_run.weights.h5"
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True),
            ModelCheckpoint(filepath=temp_weights_path, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True),
            SmartBacktrackCallback(best_weights_path=temp_weights_path, monitor_loss='val_loss', patience=4, factor=0.5, min_lr=1e-6, max_rollbacks=2)
        ]

        start_time = time.time()
        try:
            history = model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=callbacks, class_weight=class_weights_dict, verbose=1)
        except KeyboardInterrupt:
            print("\n⚠️ Multi-Run прерван пользователем.")
            break

        train_time = time.time() - start_time

        if os.path.exists(temp_weights_path):
            model.load_weights(temp_weights_path)
            os.remove(temp_weights_path)
            
        loss, acc = model.evaluate(val_dataset, verbose=0)
        print(f"🎯 Точность итерации {run}: {acc*100:.2f}%")

        if acc > global_best_acc:
            global_best_acc = acc
            global_best_run = str(run)
            save_record_model(model, history, acc, loss, train_time, run, args, seq_len, MODELS_DIR, exp_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Имя папки эксперимента")
    parser.add_argument("--runs", type=int, default=100, help="Количество циклов")
    parser.add_argument("--batch_size", type=int, default=2048, help="Размер батча")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох")
    parser.add_argument("--lr", type=float, default=1e-3, help="Стартовый Learning Rate")
    parser.add_argument("--l2_reg", type=float, default=1e-5, help="L2 регуляризация")
    args = parser.parse_args()
    main(args)