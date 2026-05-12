import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Add, Input, Dense, Dropout, Attention, LayerNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def create_model(seq_len, n_features, l2_reg=1e-5):
    policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    
    x = Dense(64, activation='linear', name='fp16_projection', dtype=policy)(inputs)
    
    # --- СВЕРТОЧНЫЙ БЛОК (Замена GRU) ---
    # Свертка 1
    conv1 = Conv1D(filters=64, kernel_size=3, padding='causal', dtype=policy)(x)
    conv1 = BatchNormalization(dtype=policy)(conv1)
    conv1 = Activation('gelu', dtype=policy)(conv1)
    conv1 = Dropout(0.2)(conv1)
    
    # Свертка 2 (с расширенным окном)
    conv2 = Conv1D(filters=64, kernel_size=5, padding='causal', dilation_rate=2, dtype=policy)(conv1)
    conv2 = BatchNormalization(dtype=policy)(conv2)
    conv2 = Activation('gelu', dtype=policy)(conv2)
    
    # Residual Connection (чтобы сеть не "забыла" исходный сигнал)
    res_x = Add(dtype=policy)([x, conv2])
    res_x = Dropout(0.2)(res_x)
    # -----------------------------------
    
    # Твой родной Attention и Pooling
    attn_out = Attention(dtype=policy)([res_x, res_x])
    x = Add(dtype=policy)([res_x, attn_out])
    x = LayerNormalization(dtype=policy)(x)
    
    avg_pool = GlobalAveragePooling1D(dtype=policy)(x)
    max_pool = GlobalMaxPooling1D(dtype=policy)(x)
    x = Concatenate(dtype=policy)([avg_pool, max_pool])
    
    x = Dense(64, kernel_regularizer=regularizers.l2(l2_reg), dtype=policy)(x)
    x = Activation('gelu', dtype=policy)(x)
    x = Dropout(0.2)(x)
    
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