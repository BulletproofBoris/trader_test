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
    Итерация 6a: Авто-экстрактор фичей (1x1 Conv) + GRU
    Цель: Очистить 68 признаков от шума ДО того, как они попадут в GRU.
    """
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    x = GaussianNoise(0.01)(inputs)
    x = LayerNormalization()(x)

    # ==========================================
    # 🧹 ФИЛЬТР ПРИЗНАКОВ (Свертка 1x1)
    # ==========================================
    # kernel_size=1 означает, что мы смотрим только на 1 день за раз.
    # filters=16 заставляет сеть сжать 68 сырых колонок в 16 самых важных.
    # Это математический аналог алгоритма PCA (Метод главных компонент), но обучаемый!
    x = Conv1D(
        filters=16, 
        kernel_size=1, 
        activation='gelu', 
        kernel_regularizer=regularizers.l2(l2_reg),
        name="feature_bottleneck"
    )(x)

    # ==========================================
    # 🧠 ВРЕМЕННОЙ АНАЛИЗ (GRU)
    # ==========================================
    # Теперь GRU дышится легко: на вход поступает всего 16 очищенных признаков,
    # и он может сосредоточиться на поиске тренда за 6 дней.
    x = GRU(
        units=32, # Память тоже можно сделать меньше, так как данные стали чище
        return_sequences=False, 
        kernel_regularizer=regularizers.l2(l2_reg),
        name="gru_temporal"
    )(x)
    x = Dropout(0.1)(x)

    # ==========================================
    # 🎯 ПРИНЯТИЕ РЕШЕНИЯ
    # ==========================================
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