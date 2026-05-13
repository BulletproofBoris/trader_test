import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def create_model(seq_len, n_features, l2_reg=1e-5):
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    
    # 0. Броня входа
    x = LayerNormalization()(inputs)
    
    # 1. Проекция (Сжимаем сырые фичи сразу до 32)
    x = Dense(32, activation='gelu', name='fp16_projection')(x)
    x = LayerNormalization()(x)
    
    # --- БЛОК 1: LSTM (Срезано с 64 до 32 нейронов) ---
    x = LSTM(
        units=32, 
        return_sequences=True, 
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = LayerNormalization()(x)
    x = Dropout(0.5)(x) # Экстремальный дропаут (отключаем половину нейронов каждую эпоху!)
    
    # --- БЛОК 2: LSTM (Срезано с 32 до 16 нейронов) ---
    x = LSTM(
        units=16, 
        return_sequences=False, 
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = LayerNormalization()(x)
    x = Dropout(0.5)(x) # Экстремальный дропаут
    
    # --- Финальный слой (Срезано с 32 до 16) ---
    x = Dense(16, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(3, activation='softmax', name='out')(x)
    
    return Model(inputs=inputs, outputs=outputs)

def save_record_model(model, history, acc, loss, train_time, run_id, dataset_name, fold, seq_len, n_features, models_dir):
    model_filename = f"trading_bot_best_acc_{acc*100:.2f}_{run_id}.keras"
    model.save(models_dir / model_filename)
    
    meta_filepath = models_dir / f"trading_bot_best_acc_{acc*100:.2f}_{run_id}_meta.json"
    with open(meta_filepath, 'w') as f:
        json.dump({
            "run_id": run_id,
            "dataset": dataset_name,
            "fold": fold,
            "seq_len": seq_len,
            "n_features": n_features,
            "best_acc": acc,
            "best_loss": loss,
            "train_time_sec": train_time,
        }, f, indent=4)