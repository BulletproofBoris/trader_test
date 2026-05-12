import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, GRU, Bidirectional, Dropout, Attention,
    Add, LayerNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def create_model(seq_len, n_features, l2_reg=1e-5):
    # Достаем глобальную политику (она уже установлена как 'mixed_float16' в главном скрипте)
    policy = tf.keras.mixed_precision.global_policy()
    
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    x = Dense(64, activation='linear', name='fp16_projection', dtype=policy)(inputs)
    
    # ВОТ ЗДЕСЬ ИСПРАВЛЕНИЕ: прокидываем policy прямо в GRU
    x = Bidirectional(
        GRU(64, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg), dtype=policy), 
        dtype=policy
    )(x)
    
    x = Dropout(0.2)(x)
    
    # И во второй слой тоже
    x = Bidirectional(
        GRU(64, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg), dtype=policy), 
        dtype=policy
    )(x)
    res_x = Dropout(0.2)(x)
    
    attn_out = Attention()([res_x, res_x])
    x = Add()([res_x, attn_out])
    x = LayerNormalization()(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    
    x = Dense(64, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Activation('gelu')(x)
    x = Dropout(0.2)(x)
    
    # Финальный слой ОСТАВЛЯЕМ в float32 (это спасает от NaN)
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