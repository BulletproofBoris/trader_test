import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, Activation, Add, Input, Dense, 
    Dropout, SpatialDropout1D, LayerNormalization, GlobalAveragePooling1D, 
    GlobalMaxPooling1D, Concatenate, MultiHeadAttention
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def create_model(seq_len, n_features, l2_reg=1e-4): # По умолчанию l2 чуть слабее, т.к. штрафуем всё
    policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    
    x = Dense(64, activation='linear', name='fp16_projection', dtype=policy)(inputs)
    
    # Общий регуляризатор для всех сверток
    reg = regularizers.l2(l2_reg)
    
    # ==========================================
    # 🧠 БЛОК 1: INCEPTION
    # ==========================================
    conv1_short = Conv1D(filters=32, kernel_size=3, padding='causal', dilation_rate=1, kernel_regularizer=reg, dtype=policy)(x)
    conv1_long = Conv1D(filters=32, kernel_size=5, padding='causal', dilation_rate=1, kernel_regularizer=reg, dtype=policy)(x)
    
    conv1 = Concatenate(dtype=policy)([conv1_short, conv1_long])
    conv1 = BatchNormalization(dtype=policy)(conv1)
    conv1 = Activation('gelu', dtype=policy)(conv1)
    # Используем SpatialDropout для временных рядов (выключает целые паттерны, а не точки)
    conv1 = SpatialDropout1D(0.3)(conv1) 
    
    # ==========================================
    # 🧠 БЛОК 2: INCEPTION + DILATION
    # ==========================================
    conv2_short = Conv1D(filters=32, kernel_size=3, padding='causal', dilation_rate=2, kernel_regularizer=reg, dtype=policy)(conv1)
    conv2_long = Conv1D(filters=32, kernel_size=5, padding='causal', dilation_rate=2, kernel_regularizer=reg, dtype=policy)(conv1)
    
    conv2 = Concatenate(dtype=policy)([conv2_short, conv2_long])
    conv2 = BatchNormalization(dtype=policy)(conv2)
    conv2 = Activation('gelu', dtype=policy)(conv2)
    conv2 = SpatialDropout1D(0.3)(conv2)

    # ==========================================
    # 🧠 БЛОК 3: DEEP TCN
    # ==========================================
    conv3 = Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=4, kernel_regularizer=reg, dtype=policy)(conv2)
    conv3 = BatchNormalization(dtype=policy)(conv3)
    conv3 = Activation('gelu', dtype=policy)(conv3)

    # Residual Connection 
    res_x = Add(dtype=policy)([x, conv3])
    res_x = SpatialDropout1D(0.3)(res_x)
    
    # ==========================================
    # 🧠 БЛОК 4: MULTI-HEAD ATTENTION (Упрощен до 2 голов, чтобы не цеплять шум)
    # ==========================================
    attn_out = MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.3, dtype=policy)(res_x, res_x)
    
    x = Add(dtype=policy)([res_x, attn_out])
    x = LayerNormalization(dtype=policy)(x)
    
    # --- ВЫХОДНОЙ БЛОК ---
    avg_pool = GlobalAveragePooling1D(dtype=policy)(x)
    max_pool = GlobalMaxPooling1D(dtype=policy)(x)
    x = Concatenate(dtype=policy)([avg_pool, max_pool])
    
    # Экстремальный Dropout(0.5) перед финальными решениями (эффект "совета директоров")
    x = Dropout(0.5)(x)
    
    x = Dense(64, kernel_regularizer=reg, dtype=policy)(x)
    x = Activation('gelu', dtype=policy)(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(3, activation='softmax', name='out', dtype='float32')(x)
    
    return Model(inputs=inputs, outputs=outputs)

def save_record_model(model, history, acc, loss, train_time, run_id, dataset_name, fold, seq_len, n_features, models_dir):
    model_filename = f"trading_bot_best_acc_{acc*100:.2f}_{run_id}.keras"
    model.save(models_dir / model_filename)
    
    meta_filepath = models_dir / f"trading_bot_best_acc_{acc*100:.2f}_{run_id}.json"
    report = {
        "model_name": model_filename,
        "run_id": str(run_id),
        "dataset": dataset_name,
        "fold": fold,
        "metrics": {"val_loss": float(loss), "val_acc": float(acc * 100)},
        "training_stats": {"training_time_seconds": float(train_time)}
    }
    with open(meta_filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)