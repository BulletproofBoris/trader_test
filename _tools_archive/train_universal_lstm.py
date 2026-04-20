# _tools/train_universal_lstm.py
import argparse
import json
import os
from pathlib import Path
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, GRU, Bidirectional, Dense, Dropout, Attention,
    Add, LayerNormalization, GlobalAveragePooling1D, LeakyReLU, Conv1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2

# --- Настройка путей и окружения ---
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_XLA'] = '1'
tf.keras.mixed_precision.set_global_policy('mixed_float16')

try:
    PROJECT_DIR = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_DIR = Path.cwd()

RESULTS_DIR = PROJECT_DIR / "2_results"
MODELS_DIR = RESULTS_DIR / "models"
TFRECORDS_DIR = RESULTS_DIR

# --- ИЗМЕНЕННЫЙ КАСТОМНЫЙ КОЛЛБЭК ---
class BacktrackReduceLROnPlateau(Callback):
    def __init__(self, best_weights_path, global_best_accuracy, monitor_loss='val_loss', monitor_acc='val_accuracy',
                 factor=0.1, patience=5, verbose=1, min_delta=1e-4, min_lr=0, threshold=0.02):
        super(BacktrackReduceLROnPlateau, self).__init__()
        self.monitor_loss = monitor_loss
        self.monitor_acc = monitor_acc
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.best_weights_path = str(best_weights_path)
        self.global_best_accuracy = global_best_accuracy
        self.threshold = threshold

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_loss = np.inf
        self.best_accuracy = -np.inf
        self.best_epoch = 0  # <-- НОВОЕ: Запоминаем лучшую эпоху

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor_loss)
        current_accuracy = logs.get(self.monitor_acc)
        if current_loss is None or current_accuracy is None:
            return

        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_epoch = epoch + 1  # <-- НОВОЕ: Обновляем лучшую эпоху
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                is_hopeless = self.best_accuracy < self.global_best_accuracy * (1 - self.threshold)
                if is_hopeless and self.global_best_accuracy > 0:
                    if self.verbose > 0:
                        print(f"\nEpoch {epoch + 1}: Лучшая точность этого запуска ({self.best_accuracy:.4f}) "
                              f"значительно хуже глобального рекорда ({self.global_best_accuracy:.4f}). Прерываю.")
                    self.model.stop_training = True
                    return

                if os.path.exists(self.best_weights_path):
                    if self.verbose > 0:
                        # <-- НОВОЕ: Улучшенное сообщение
                        print(f"\nEpoch {epoch + 1}: {self.monitor_loss} не улучшался {self.patience} эпох. "
                              f"Восстанавливаю лучшие веса из эпохи #{self.best_epoch}...")
                    self.model.load_weights(self.best_weights_path)

                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    self.model.optimizer.learning_rate.assign(new_lr)
                    if self.verbose > 0:
                        print(f'Epoch {epoch + 1}: Снижаю скорость обучения до {new_lr:.0e}.')
                    self.wait = 0

# --- Функции для работы с данными ---
def parse_tfrecord_fn(example, lookback, n_features):
    feature_description = {'sequence': tf.io.FixedLenFeature([], tf.string), 'target': tf.io.FixedLenFeature([], tf.int64)}
    example = tf.io.parse_single_example(example, feature_description)
    sequence = tf.io.parse_tensor(example['sequence'], out_type=tf.float32)
    label = example['target']
    sequence.set_shape([lookback, n_features])
    label = tf.one_hot(label, depth=3)
    label.set_shape([3])
    return sequence, label

def load_tfrecord_dataset(file_path, batch_size, lookback, n_features):
    dataset = tf.data.TFRecordDataset(str(file_path))
    dataset = dataset.map(lambda x: parse_tfrecord_fn(x, lookback, n_features), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- Старая архитектура ---
def create_lstm_model(seq_len, n_features, l2_reg, leaky_relu_negative_slope):
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    x = Bidirectional(GRU(6, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg)))(inputs)
    x = Dropout(0.0258)(x)
    x = Bidirectional(GRU(7, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg)))(x)
    res_x = Dropout(0.0258)(x)
    attn_out = Attention()([res_x, res_x])
    x = Add()([res_x, attn_out])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = LeakyReLU(alpha=leaky_relu_negative_slope)(x)
    x = Dropout(0.0258)(x)
    outputs = Dense(3, activation='softmax', name='out', dtype='float32')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- Новая архитектура из тюнера ---
def create_model_from_params(params: dict, seq_len: int, n_features: int) -> Model:
    inputs = Input(shape=(seq_len, n_features))
    x = Conv1D(filters=params['conv_filters'], kernel_size=params['kernel_size'], padding='causal', kernel_regularizer=l2(params['l2_reg']))(inputs)
    x = LayerNormalization()(x)
    x = LeakyReLU(params['lrelu_neg_slope'])(x)
    x = Dropout(params['dropout_conv'])(x)
    x = Bidirectional(GRU(params['gru_units_1'], return_sequences=True, kernel_regularizer=l2(params['l2_reg'])))(x)
    x = Dropout(params['dropout_gru'])(x)
    x = Bidirectional(GRU(params['gru_units_2'], return_sequences=True, kernel_regularizer=l2(params['l2_reg'])))(x)
    res_x = Dropout(params['dropout_gru'])(x)
    attn_out = Attention()([res_x, res_x])
    x = Add()([res_x, attn_out])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(params['dense_units'], kernel_regularizer=l2(params['l2_reg']))(x)
    x = LeakyReLU(params['lrelu_neg_slope'])(x)
    x = Dropout(params['dropout_gru'])(x)
    outputs = Dense(3, activation='softmax', dtype='float32')(x)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Основная логика ---
def main(args):
    model_name = args.model_name
    data_model_name = args.data_model_name if args.data_model_name else model_name
    
    print(f"Использую данные от модели: '{data_model_name}'")
    print(f"Обученная модель будет сохранена как: '{model_name}'")

    output_dir = MODELS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tfrecord_base_path = TFRECORDS_DIR / f"tfrecord_{data_model_name}"
    meta_path = tfrecord_base_path / "metadata.json"
    
    try:
        with open(meta_path, 'r') as f: metadata = json.load(f)
        print("✅ Метаданные загружены.")
    except FileNotFoundError as e:
        print(f"❌ Ошибка: Необходимый файл не найден - {e}."); return

    best_params = None
    if args.params_from_tuner:
        print(f"\n---  режим '--params_from_tuner' ---")
        if args.params_file:
            params_path = RESULTS_DIR / args.params_file
            print(f"Использую указанный файл с параметрами: {params_path}")
        else:
            params_path = RESULTS_DIR / f"best_params_{model_name}.json"
            print(f"Использую файл с параметрами по умолчанию: {params_path}")
        try:
            with open(params_path, 'r') as f: best_params = json.load(f)
            print(f"✅ Лучшие параметры успешно загружены."); print(json.dumps(best_params, indent=2))
        except FileNotFoundError:
            print(f"❌ Ошибка: Файл с параметрами не найден: {params_path}"); return
    else:
        print("\n--- Режим 'ручных' параметров (из командной строки) ---")

    seq_len = metadata['lookback']
    n_features = metadata['n_features']
    class_weights_dict = {int(k): v for k, v in metadata['class_weights'].items()}
    
    print("\nЗагружаю датасеты TFRecord...")
    train_dataset = load_tfrecord_dataset(tfrecord_base_path / "train.tfrecord", args.batch_size, seq_len, n_features)
    val_dataset = load_tfrecord_dataset(tfrecord_base_path / "test.tfrecord", args.batch_size, seq_len, n_features)
    print("✅ Датасеты TFRecord готовы.")

    global_best_val_accuracy = 0.0
    best_score_path = output_dir / "best_val_accuracy.json"
    if best_score_path.exists():
        try:
            with open(best_score_path, 'r') as f:
                global_best_val_accuracy = json.load(f).get("val_accuracy", 0.0)
        except json.JSONDecodeError: pass
    
    print(f"Текущий глобальный рекорд val_accuracy: {global_best_val_accuracy:.5f}")
    print(f"--- Запуск {args.runs} независимых сеансов обучения ---")

    for i in range(1, args.runs + 1):
        print(f"\n========================= Начинаю запуск №{i}/{args.runs} =========================")
        tf.keras.backend.clear_session()
        
        if best_params:
            model = create_model_from_params(best_params, seq_len, n_features)
        else:
            model = create_lstm_model(seq_len, n_features, args.l2_reg, args.leaky_relu_negative_slope)
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        if i == 1: model.summary()

        temp_weights_path = output_dir / f"temp_best_run_{i}.weights.h5"

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, verbose=1),
            ModelCheckpoint(filepath=temp_weights_path, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True),
            BacktrackReduceLROnPlateau(
                best_weights_path=temp_weights_path, global_best_accuracy=global_best_val_accuracy,
                monitor_loss='val_loss', monitor_acc='val_accuracy', patience=6, factor=0.2,
                min_lr=5e-4 if not best_params else 1e-6, verbose=1
            )
        ]

        history = model.fit(train_dataset, epochs=150, validation_data=val_dataset, callbacks=callbacks, class_weight=class_weights_dict, verbose=args.verbose)
        
        if not history.history or not history.history.get('val_loss'):
            print("--- ⚠️ Обучение было прервано или завершилось с ошибкой. Пропускаю запуск. ---")
            if os.path.exists(temp_weights_path): os.remove(temp_weights_path)
            continue

        if os.path.exists(temp_weights_path):
            print("Загружаю лучшие веса, найденные в этом запуске...")
            model.load_weights(temp_weights_path)

        results = model.evaluate(val_dataset, verbose=0)
        iteration_best_val_accuracy = results[1]

        print(f"\n--- Результаты запуска {i} ---")
        print(f"  Лучшая val_accuracy этого запуска: {iteration_best_val_accuracy:.5f}")

        if iteration_best_val_accuracy > global_best_val_accuracy:
            print(f"🏆 Новый ГЛОБАЛЬНЫЙ рекорд! ({iteration_best_val_accuracy:.5f} > {global_best_val_accuracy:.5f}).")
            global_best_val_accuracy = iteration_best_val_accuracy
            model.save(output_dir / "model.keras")
            with open(best_score_path, 'w') as f:
                json.dump({"val_accuracy": global_best_val_accuracy}, f)
        else:
            print(f"Результат не превысил глобальный рекорд ({global_best_val_accuracy:.5f}).")
        
        if os.path.exists(temp_weights_path):
            os.remove(temp_weights_path)
            
    print(f"\n✅ Все запуски завершены. Финальная лучшая модель сохранена в: {output_dir / 'model.keras'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение N независимых моделей и выбор лучшей.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_model_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--params_from_tuner", action="store_true", help="Загрузить лучшие параметры из JSON-файла тюнера.")
    parser.add_argument("--params_file", type=str, default=None, help="Точное имя файла с параметрами в папке 2_results.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=1e-5)
    parser.add_argument("--leaky_relu_negative_slope", type=float, default=0.2)
    
    args, _ = parser.parse_known_args()
    main(args)