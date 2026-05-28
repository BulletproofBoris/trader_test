import json
from pathlib import Path
import math
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization, Conv1D,
    MultiHeadAttention, Add, GaussianNoise, GRU, Multiply, GlobalAveragePooling1D, Reshape, SpatialDropout1D
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def create_model(seq_len, n_features, l2_reg):
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    
    # Легкий шум не дает сети выучить наизусть конкретные значения цен
    x = GaussianNoise(0.01)(inputs)
    x = LayerNormalization()(x)

    # 1. Bottleneck: "Умный фильтр"
    # Мгновенно собирает 68 сырых признаков в 16 плотных мета-факторов на каждом баре.
    x = Conv1D(
        filters=32, 
        kernel_size=1, 
        activation='gelu', 
        kernel_regularizer=regularizers.l2(l2_reg),
        name="feature_bottleneck"
    )(x)

    x = SpatialDropout1D(0.05)(x)

    # 2. GRU: Снайперский выстрел
    # return_sequences=False заставляет GRU "молчать" первые 5 дней 
    # и выдать всю накопленную уверенность строго на 6-м (последнем) баре.
    x = GRU(
        units=32, 
        return_sequences=False, 
        kernel_regularizer=regularizers.l2(l2_reg),
        name="gru_temporal"
    )(x)
    x = Dropout(0.1)(x)

    # 3. Финальный классификатор
    x = Dense(32, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(0.1)(x)
    
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