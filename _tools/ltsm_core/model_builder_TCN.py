import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LayerNormalization, Conv1D, GlobalAveragePooling1D, 
    Concatenate, Dense, Dropout, Add, Lambda, Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def create_model(seq_len, n_features, l2_reg):
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    x = LayerNormalization()(inputs)

    # Выравниваем размерность каналов для остаточных связей
    filters = 32
    res_x = Dense(filters, activation='linear')(x)

    # ==========================================
    # 🌊 ВЕТКА 1: TCN (Dilated Causal Convolutions)
    # ==========================================
    # Экспоненциальное расширение окна (1, 2, 4, 8, 16)
    # Это позволяет сети охватить все 90 дней без потери деталей
    for dilation_rate in [1, 2, 4, 8, 16]:
        # Causal padding гарантирует, что мы не заглядываем в будущее
        conv = Conv1D(filters=filters, 
                      kernel_size=5, 
                      padding='causal', 
                      dilation_rate=dilation_rate, 
                      activation='gelu',
                      kernel_regularizer=regularizers.l2(l2_reg))(res_x)
        conv = Dropout(0.2)(conv)
        
        # Свертка с ядром 1 для выравнивания размерностей (если нужно)
        # В данном случае каналы уже равны (32), так что просто складываем
        res_x = Add()([res_x, conv]) # ⚡ Residual Connection

    # Извлекаем финальный "скелет" тренда
    tcn_out = GlobalAveragePooling1D()(res_x)

    # ==========================================
    # 🔥 ВЕТКА 2: МИКРО-КОНТЕКСТ (Последние 15 дней)
    # ==========================================
    # Оставляем эту концепцию, она математически верна для горизонта 15
    recent_data = Lambda(lambda s: s[:, -15:, :])(x)
    recent_context = Flatten()(recent_data)
    recent_context = Dense(32, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(recent_context)
    recent_context = Dropout(0.1)(recent_context)

    # ==========================================
    # 🧬 СЛИЯНИЕ И ФИНАЛ
    # ==========================================
    merged = Concatenate()([tcn_out, recent_context])
    
    # Финальный классификатор
    dense = Dense(32, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(merged)
    dense = Dropout(0.2)(dense)
    
    outputs = Dense(3, activation='softmax', name='out')(dense)
    
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