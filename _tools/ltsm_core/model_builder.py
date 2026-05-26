import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, LSTM, Flatten, Activation, Dot, Conv1D, GlobalAveragePooling1D, Concatenate,Reshape, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import math

def se_block(input_tensor, reduction=4):
    channels = input_tensor.shape[-1]
    # Squeeze
    x = GlobalAveragePooling1D()(input_tensor)
    x = Reshape((1, channels))(x)
    # Excitation
    x = Dense(channels // reduction, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(channels, activation='sigmoid', kernel_initializer='he_normal')(x)
    # Scale
    return Multiply()([input_tensor, x])

def create_model(seq_len, n_features, l2_reg):
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    
    # Нормализуем сырой вход
    x = LayerNormalization()(inputs)

    # ==========================================
    # ⚡ ВЕТКА 1: АДАПТИВНЫЙ ИМПУЛЬС (CNN)
    # ==========================================
    p = 1
    f = 12
    kernel_1 = max(1, math.ceil(seq_len * p * 1))
    kernel_2 = max(1, math.ceil(seq_len * p * 3))
    kernel_3 = max(1, math.ceil(seq_len * p * 5))
    kernel_4 = max(1, math.ceil(seq_len * p * 7))
    
    # Убрали лишние LayerNorm, добавили Dropout для регуляризации фильтров
    conv_1 = Conv1D(filters=f, kernel_size=kernel_1, padding='same', activation='relu')(x)
    conv_2 = Conv1D(filters=f, kernel_size=kernel_2, padding='same', activation='relu')(x)

    conv_3 = Conv1D(filters=f, kernel_size=kernel_3, padding='same', activation='relu')(conv_1)
    # Используем AveragePooling вместо Max, чтобы сохранить усредненный фон паттерна, а не выброс
    conv_3_pool = GlobalAveragePooling1D()(conv_3) 

    conv_4 = Conv1D(filters=f, kernel_size=kernel_4, padding='same', activation='relu')(conv_2)
    conv_4_pool = GlobalAveragePooling1D()(conv_4)

    # Вектор 24
    conv_out = Concatenate()([conv_3_pool, conv_4_pool]) 

    # ==========================================
    # 🧠 ВЕТКА 2: ГЛУБОКИЙ КОНТЕКСТ (LSTM + Attention)
    # ==========================================
    lstm = LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(x)
    lstm = Dropout(0.1)(lstm)

    lstm_out = LSTM(16, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(lstm)
    lstm_out = Dropout(0.1)(lstm_out)

    # ИСПРАВЛЕННЫЙ ATTENTION: linear вместо tanh!
    attention_scores = Dense(1, activation='linear')(lstm_out)
    attention_scores = Flatten()(attention_scores)
    attention_weights = Activation('softmax', name='attention_weights')(attention_scores)
    
    # Вектор 16 (динамически взвешенный по времени)
    lstm_att = Dot(axes=1)([attention_weights, lstm_out])

    # ==========================================
    # 🧬 СЛИЯНИЕ И ФИНАЛ
    # ==========================================
    # Склеиваем ветки (24 + 16 = 40 признаков)
    merged = Concatenate()([conv_out, lstm_att])
    
    # Здесь LayerNorm уместен, чтобы выровнять масштабы выходов CNN и LSTM перед финальным Dense
    merged = LayerNormalization()(merged)
    
    x = Dense(32, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(merged)
    x = Dropout(0.4)(x)
    
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