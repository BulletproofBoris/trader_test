# _tools/hyperparameter_tuner.py
import os
import warnings

os.environ['TF_ENABLE_XLA'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import optuna
from optuna.pruners import HyperbandPruner
from optuna.integration import TFKerasPruningCallback

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, GRU, Bidirectional, Dense, Dropout, Attention,
    Add, LayerNormalization, GlobalAveragePooling1D, LeakyReLU, Conv1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Включаем смешанную точность для теста на tf-nightly
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- Пути ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()

RESULTS_DIR = PROJECT_ROOT / "2_results"
TFRECORDS_DIR = RESULTS_DIR

# --- НОВЫЙ КОЛЛБЭК ДЛЯ СОХРАНЕНИЯ ЛУЧШИХ ПАРАМЕТРОВ ---
def save_best_trial_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    """
    Callback для сохранения лучших гиперпараметров в JSON файл после каждой итерации,
    если текущая итерация показала лучший результат.
    """
    if study.best_trial.number == trial.number:
        print(f"\n[Callback] Найден новый лучший результат (trial #{trial.number}): "
              f"val_accuracy = {trial.value:.5f}. Сохраняю параметры.")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        params_path = RESULTS_DIR / f"best_params_{study.study_name}.json"
        best_params = study.best_trial.params

        try:
            with open(params_path, 'w') as f:
                json.dump(best_params, f, indent=4)
            print(f"[Callback] ✅ Лучшие параметры сохранены в: {params_path}")
        except Exception as e:
            print(f"[Callback] ❌ Ошибка при сохранении параметров: {e}")
# --- КОНЕЦ НОВОГО БЛОКА ---


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
    dataset = dataset.cache()
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- Функция создания модели ---
def create_tuned_model(seq_len, n_features, trial):
    conv_filters = trial.suggest_categorical("conv_filters", [42, 60, 72, 88, 104, 120, 136, 152])
    kernel_size = trial.suggest_categorical("kernel_size", [5, 6, 7, 8, 9, 10, 11])
    gru_units_1 = trial.suggest_categorical("gru_units_1", [6, 7, 8, 9, 10, 11, 12, 13])
    gru_units_2 = trial.suggest_categorical("gru_units_2", [7, 8, 9, 10, 11, 12, 13, 14])
    dense_units = trial.suggest_categorical("dense_units", [32, 36, 40, 44, 48, 52])

    dropout_conv = trial.suggest_float("dropout_conv", 0.03, 0.2)
    dropout_gru = trial.suggest_float("dropout_gru", 0.01, 0.2)
    l2_reg = trial.suggest_float("l2_reg", 1e-8, 1e-5, log=True)
    alpha = trial.suggest_float("lrelu_neg_slope", 0.01, 0.4)

    inputs = Input(shape=(seq_len, n_features))

    x = Conv1D(filters=conv_filters, kernel_size=kernel_size, padding='causal', kernel_regularizer=l2(l2_reg))(inputs)
    x = LayerNormalization()(x)
    x = LeakyReLU(alpha)(x)
    x = Dropout(dropout_conv)(x)

    x = Bidirectional(GRU(gru_units_1, return_sequences=True, kernel_regularizer=l2(l2_reg)))(x)
    x = Dropout(dropout_gru)(x)
    x = Bidirectional(GRU(gru_units_2, return_sequences=True, kernel_regularizer=l2(l2_reg)))(x)
    res_x = Dropout(dropout_gru)(x)

    attn_out = Attention()([res_x, res_x])
    x = Add()([res_x, attn_out])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)

    x = Dense(dense_units, kernel_regularizer=l2(l2_reg))(x)
    x = LeakyReLU(alpha)(x)
    x = Dropout(dropout_gru)(x)

    outputs = Dense(3, activation='softmax', dtype='float32')(x)

    return Model(inputs=inputs, outputs=outputs)

# --- Главная функция "цели" для Optuna ---
def objective(trial, args, metadata, train_dataset, val_dataset):
    tf.keras.backend.clear_session()

    model = create_tuned_model(metadata['lookback'], metadata['n_features'], trial)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    pruning_callback = TFKerasPruningCallback(trial, 'val_accuracy')
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=12, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=0
    )
    callbacks = [early_stopping, pruning_callback, reduce_lr]

    history = model.fit(
        train_dataset,
        epochs=70,
        validation_data=val_dataset,
        callbacks=callbacks,
        class_weight={int(k): v for k, v in metadata['class_weights'].items()},
        verbose=0
    )

    best_epoch_index = np.argmax(history.history.get('val_accuracy', [0]))
    best_val_accuracy = history.history['val_accuracy'][best_epoch_index]
    best_val_loss = history.history['val_loss'][best_epoch_index]

    trial.set_user_attr("val_loss", best_val_loss)

    return best_val_accuracy

def print_trial_callback(study, trial):
    is_pruned = "PRUNED" if trial.state == optuna.trial.TrialState.PRUNED else ""
    current_val_acc = trial.value
    if current_val_acc is None:
        current_val_acc = float('nan')

    current_val_loss = trial.user_attrs.get('val_loss', float('nan'))

    best_trial = study.best_trial
    best_val_acc = best_trial.value
    best_val_loss = best_trial.user_attrs.get('val_loss', float('nan'))

    print(f"Trial {trial.number:3d} finished. "
          f"Current: [Acc: {current_val_acc:.4f}, Loss: {current_val_loss:.4f}]. "
          f"Best (Trial #{best_trial.number}): [Acc: {best_val_acc:.4f}, Loss: {best_val_loss:.4f}]. {is_pruned}")

# --- Основная логика ---
def main(args):
    tfrecord_base_path = TFRECORDS_DIR / f"tfrecord_{args.model_name}"
    meta_path = tfrecord_base_path / "metadata.json"

    try:
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"❌ Ошибка: Не найден файл метаданных: {meta_path}")
        return

    train_dataset = load_tfrecord_dataset(tfrecord_base_path / "train.tfrecord", args.batch_size, metadata['lookback'], metadata['n_features'])
    val_dataset = load_tfrecord_dataset(tfrecord_base_path / "test.tfrecord", args.batch_size, metadata['lookback'], metadata['n_features'])

    pruner = HyperbandPruner()
    # ---> ИЗМЕНЕНИЕ: Добавляем study_name <---
    study = optuna.create_study(
        direction='maximize',
        pruner=pruner,
        study_name=args.model_name
    )

    # ---> ИЗМЕНЕНИЕ: Добавляем наш новый callback для сохранения <---
    study.optimize(
        lambda trial: objective(trial, args, metadata, train_dataset, val_dataset),
        n_trials=args.n_trials,
        show_progress_bar=False,
        callbacks=[print_trial_callback, save_best_trial_callback]
    )

    print("\n--- 🏆 Поиск завершен! ---")

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    print(f"Статистика: {len(complete_trials)} завершено, {len(pruned_trials)} отсечено (pruned).")

    best_trial = study.best_trial

    print("\n--- 🥇 Лучший результат по максимальной точности ---")
    print(f"Trial #{best_trial.number}")

    best_val_loss = best_trial.user_attrs.get('val_loss', 'N/A')
    print(f"  - val_loss: {best_val_loss:.4f}")
    print(f"  - val_accuracy: {best_trial.value:.4f}")
    print("Найденные оптимальные параметры:")
    print(json.dumps(best_trial.params, indent=4))

    params_path = RESULTS_DIR / f"best_params_{args.model_name}.json"
    with open(params_path, 'w') as f:
        json.dump(best_trial.params, f, indent=4)
    print(f"\n✅ Лучшие параметры (по точности) сохранены в: {params_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Многоцелевой поиск гиперпараметров с помощью Optuna.")
    parser.add_argument('--model_name', type=str, required=True, help="Имя модели, для которой подбираем параметры.")
    parser.add_argument('--n_trials', type=int, default=200, help="Количество попыток (обучений).")
    parser.add_argument('--batch_size', type=int, default=4096)

    args, _ = parser.parse_known_args()
    main(args)