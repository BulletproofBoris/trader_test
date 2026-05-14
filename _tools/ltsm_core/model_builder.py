import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, LSTM, Flatten, Activation, Dot, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def create_model(seq_len, n_features, l2_reg=1e-4): # Чуть повысили базовый L2
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    
    # Нормализуем входы
    x = LayerNormalization()(inputs)
    
    # 🛡️ АНТИ-ЗУБРЕЖКА: Добавляем 2% случайного шума
    # Работает только на Train! На Val/Test сеть будет видеть чистые данные
    x = GaussianNoise(0.01)(x)

    # --- БЛОК 1 (32 нейрона) ---
    x = LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    # --- БЛОК 2 (16 нейронов) ---
    lstm_out = LSTM(16, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(x)
    lstm_out = LayerNormalization()(lstm_out)
    lstm_out = Dropout(0.3)(lstm_out)

    # 👁️ МЕХАНИЗМ ВНИМАНИЯ (Attention)
    attention_scores = Dense(1, activation='tanh')(lstm_out)
    attention_scores = Flatten()(attention_scores)
    attention_weights = Activation('softmax', name='attention_weights')(attention_scores)
    x = Dot(axes=1)([attention_weights, lstm_out])
    
    # --- Финальный слой (Расширен до 16) ---
    x = Dense(16, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
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