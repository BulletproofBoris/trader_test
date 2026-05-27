import json
from pathlib import Path
import math
import tensorflow as tf
from tensorflow.keras.layers import Input, LayerNormalization, Conv1D, GlobalAveragePooling1D, GaussianNoise, GRU
from tensorflow.keras.layers import GlobalMaxPooling1D, Concatenate, LSTM, Dropout, Dense, Flatten, Activation, Dot, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def create_model(seq_len, n_features, l2_reg):
    """
    Итерация 3: Рекуррентная сеть (GRU)
    Цель: Обработать 6 дней последовательно, сделав максимальный акцент на последние дни.
    """
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    
    # 1. Защита от зубрежки (оставляем, так как фичей много - ~68)
    x = GaussianNoise(0.01)(inputs)
    x = LayerNormalization()(x)

    # 2. Чтение последовательности через GRU
    # return_sequences=False означает, что GRU прочитает все 6 дней, 
    # но выдаст нам только один финальный вектор (вывод после прочтения 6-го дня).
    # 64 юнита - достаточное сжатие для 68 фичей.
    x = GRU(
        units=64, 
        return_sequences=False, 
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = Dropout(0.2)(x)

    # 3. Принятие решения
    x = Dense(32, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(0.2)(x)
    
    # 4. Выход на 3 класса
    outputs = Dense(3, activation='softmax', name='out')(x)
    
    return Model(inputs=inputs, outputs=outputs)

def save_record_model(model, history, acc, loss, train_time, run_id, dataset_name, fold, seq_len, n_features, models_dir):
    # 1. Меняем фокус в названии файлов: Сначала LOSS, потом ACC
    model_filename = f"trading_bot_loss_{loss:.4f}_acc_{acc*100:.2f}_{run_id}.keras"
    meta_filename = f"trading_bot_loss_{loss:.4f}_acc_{acc*100:.2f}_{run_id}.json"
    
    model_path = models_dir / model_filename
    meta_path = models_dir / meta_filename
    
    # 2. Сохраняем модель
    model.save(model_path)
    
    # 3. Сохраняем метаданные в ИДЕАЛЬНОМ формате, который ждет clean_lstm_models.py
    meta_data = {
        "model_name": model_filename,
        "run_id": run_id,
        "dataset": dataset_name,
        "fold": fold,
        "seq_len": seq_len,
        "n_features": n_features,
        "metrics": {
            "val_loss": float(loss),
            "val_acc": float(acc * 100.0)
        },
        "training_stats": {
            "training_time_seconds": float(train_time)
        }
    }
    
    with open(meta_path, 'w', encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4, ensure_ascii=False)