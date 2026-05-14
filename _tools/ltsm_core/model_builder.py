import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, LSTM, Flatten, Activation, Dot, Conv1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def create_model(seq_len, n_features, l2_reg=1e-5): # <-- Вернули L2 к адекватному 1e-5
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    x = LayerNormalization()(inputs)

    # ==========================================
    # ⚡ ВЕТКА 1: Быстрый импульс (CNN)
    # Ищет локальные ценовые паттерны (свечные формации)
    # ==========================================
    conv = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    conv = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(conv)
    # Выхватываем самые сильные сигналы за все 90 дней (вектор 32)
    conv_out = GlobalMaxPooling1D()(conv) 

    # ==========================================
    # 🧠 ВЕТКА 2: Глубокий контекст (LSTM + Attention)
    # Анализирует макро-тренд и взаимосвязи
    # ==========================================
    lstm = LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(x)
    lstm = LayerNormalization()(lstm)
    lstm = Dropout(0.2)(lstm) # Снизили Dropout до 0.2

    lstm_out = LSTM(16, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(lstm)
    lstm_out = LayerNormalization()(lstm_out)
    
    # Внимание только на LSTM-ветке
    attention_scores = Dense(1, activation='tanh')(lstm_out)
    attention_scores = Flatten()(attention_scores)
    attention_weights = Activation('softmax', name='attention_weights')(attention_scores)
    lstm_att = Dot(axes=1)([attention_weights, lstm_out]) # Вектор 16

    # ==========================================
    # 🧬 СЛИЯНИЕ И ФИНАЛ
    # ==========================================
    # Соединяем сигналы CNN (32) и LSTM (16) = 48 признаков
    merged = Concatenate()([conv_out, lstm_att])
    
    x = Dense(32, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(merged)
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