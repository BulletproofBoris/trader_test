import json
from pathlib import Path
import math
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Conv1D, GaussianNoise, GRU, Multiply, GlobalAveragePooling1D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def create_model(seq_len, n_features, l2_reg):
    """
    Итерация 8: 1x1 Conv + SE-Блок + GRU
    Цель: Научить сеть динамически "фокусироваться" на самых важных 
    мета-признаках перед тем, как отдать их в GRU.
    """
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    x = GaussianNoise(0.01)(inputs)
    x = LayerNormalization()(x)

    # 1. ФИЛЬТР ПРИЗНАКОВ (Bottleneck)
    # Сжимаем 68 фичей в 16 чистых сигналов
    encoded = Conv1D(
        filters=16, 
        kernel_size=1, 
        activation='gelu', 
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)

    # ==========================================
    # 🌟 SQUEEZE-AND-EXCITATION (SE) БЛОК 
    # ==========================================
    # Squeeze: Сжимаем каждый из 16 каналов в одно число (оцениваем его общую силу)
    se = GlobalAveragePooling1D()(encoded)
    
    # Excitation: Пропускаем через маленькую нейросеть, чтобы вычислить веса для каждого канала
    se = Dense(8, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(se)
    se = Dense(16, activation='sigmoid')(se) # Sigmoid даст проценты от 0 до 1 (громкость канала)
    
    # Меняем форму, чтобы можно было умножить на наши данные
    se = Reshape((1, 16))(se)
    
    # Применяем "эквалайзер": умножаем данные на вычисленную громкость
    encoded_se = Multiply()([encoded, se])
    # ==========================================

    # 2. ВРЕМЕННОЙ АНАЛИЗ (GRU)
    # Теперь GRU получает не просто 16 признаков, а 16 признаков с ПРАВИЛЬНОЙ громкостью
    x = GRU(
        units=32, 
        return_sequences=False, 
        kernel_regularizer=regularizers.l2(l2_reg)
    )(encoded_se)
    
    x = Dropout(0.1)(x) # Держим Dropout низким, данные очень чистые!

    # 3. ФИНАЛЬНЫЙ ВЫВОД
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