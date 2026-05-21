import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, LSTM, Flatten, Activation, Dot, Conv1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import math

def create_model(seq_len, n_features, l2_reg):
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    x = LayerNormalization()(inputs)

    # ==========================================
    # ⚡ ВЕТКА 1: АДАПТИВНЫЙ ИМПУЛЬС (CNN)
    # ==========================================
    # Вычисляем размеры ядер динамически.
    kernel_1 = max(1, math.ceil(seq_len * 0.1))
    kernel_2 = max(1, math.ceil(seq_len * 0.2))
    kernel_3 = max(1, math.ceil(seq_len * 0.3))
    kernel_4 = max(1, math.ceil(seq_len * 0.4))
    kernel_5 = max(1, math.ceil(seq_len * 0.5))
    kernel_6 = max(1, math.ceil(seq_len * 0.6))
    
    # Используем параллельные свертки (Inception-style), а не последовательные,
    # чтобы короткие и длинные паттерны не "перемешивались" раньше времени.
    conv_1 = Conv1D(filters=8, kernel_size=kernel_1, padding='same', activation='relu')(x)
    conv_1_pool = GlobalMaxPooling1D()(conv_1)
    
    conv_2 = Conv1D(filters=8, kernel_size=kernel_2, padding='same', activation='relu')(x)
    conv_2_pool = GlobalMaxPooling1D()(conv_2)

    conv_3 = Conv1D(filters=8, kernel_size=kernel_3, padding='same', activation='relu')(x)
    conv_3_pool = GlobalMaxPooling1D()(conv_3)
    
    conv_4 = Conv1D(filters=8, kernel_size=kernel_4, padding='same', activation='relu')(conv_1)
    conv_4_pool = GlobalMaxPooling1D()(conv_4)

    conv_5 = Conv1D(filters=8, kernel_size=kernel_5, padding='same', activation='relu')(conv_2)
    conv_5_pool = GlobalMaxPooling1D()(conv_5)

    conv_6 = Conv1D(filters=8, kernel_size=kernel_6, padding='same', activation='relu')(conv_3)
    conv_6_pool = GlobalMaxPooling1D()(conv_6)

    # Склеиваем оба масштаба
    conv_out = Concatenate()([conv_4_pool,conv_5_pool,conv_6_pool]) # Вектор 8*3=24

    # ==========================================
    # 🧠 ВЕТКА 2: ГЛУБОКИЙ КОНТЕКСТ (LSTM + Attention)
    # ==========================================
    lstm = LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(x)
    lstm = LayerNormalization()(lstm)
    lstm = Dropout(0.2)(lstm)

    lstm_out = LSTM(16, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(lstm)
    lstm_out = LayerNormalization()(lstm_out)
    
    attention_scores = Dense(1, activation='tanh')(lstm_out)
    attention_scores = Flatten()(attention_scores)
    attention_weights = Activation('softmax', name='attention_weights')(attention_scores)
    lstm_att = Dot(axes=1)([attention_weights, lstm_out]) # Вектор 16

    # ==========================================
    # 🧬 СЛИЯНИЕ И ФИНАЛ
    # ==========================================
    # Соединяем сигналы адаптивной CNN (64) и LSTM (16) = 80 признаков
    merged = Concatenate()([conv_out, lstm_att])
    
    # Слегка увеличим плотный слой, чтобы переварить 80 признаков
    x = Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(merged)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
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