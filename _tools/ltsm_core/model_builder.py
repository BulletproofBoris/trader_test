import json
from pathlib import Path
import math
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization, Reshape, Concatenate,
                                     Conv1D, GaussianNoise, GRU, GlobalMaxPooling1D, Multiply,
                                     Add, Activation, Flatten, GlobalAveragePooling1D, SpatialDropout1D)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

# ==========================================
# 1. АРХИТЕКТУРА С РЕКУРСИЕЙ (conv1d+gru) - Твой чемпион
# ==========================================
def _create_conv1d_gru_model(seq_len, n_features, l2_reg):
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    x = GaussianNoise(0.01)(inputs)
    x = LayerNormalization()(x)

    x = Conv1D(filters=16, kernel_size=1, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg), name="feature_bottleneck")(x)
    
    x = GRU(units=32, return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg), name="gru_temporal")(x)
    x = Dropout(0.1)(x)

    x = Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(0.1)(x)
    
    outputs = Dense(3, activation='softmax', name='out')(x)
    return Model(inputs=inputs, outputs=outputs)

# ==========================================
# 2. АРХИТЕКТУРА БЕЗ РЕКУРСИИ (cnn)
# ==========================================
def squeeze_and_excitation_block(x, ratio=4):
    """SE-блок: Динамическое внимание к каналам (признакам)"""
    filters = x.shape[-1]
    
    # Squeeze: сжимаем время (6 дней -> 1 число на каждый фильтр)
    se = GlobalAveragePooling1D()(x)
    
    # Excite: вычисляем веса важности для каждого фильтра
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(se)
    
    # Умножаем веса на исходный тензор
    se = Reshape((1, filters))(se)
    return Multiply()([x, se])

def inception_residual_block(x, l2_reg):
    """Inception + Residual: Ищет паттерны разной длины (1, 2, 3, 5 дней)"""
    shortcut = x
    
    # Ветка 1: Точечные паттерны (1 день - प्राइस экшен)
    branch1 = Conv1D(16, kernel_size=1, padding='causal', activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    # Ветка 2: Микро-паттерны (2 дня - поглощения)
    branch2 = Conv1D(16, kernel_size=2, padding='causal', activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    # Ветка 3: Стандартные паттерны (3 дня)
    branch3 = Conv1D(16, kernel_size=3, padding='causal', activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)

    # Ветка 4: Макро-паттерны (5 дней - тренд недели)
    branch5 = Conv1D(16, kernel_size=5, padding='causal', activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    # Склеиваем все 4 ветки (получаем 16*4 = 64 фильтра)
    x = Concatenate(axis=-1)([branch1, branch2, branch3, branch5])
    x = LayerNormalization()(x)
    
    # Внимание! Сеть сама решает, какая ветка (2, 3 или 5 дней) сейчас важнее
    x = squeeze_and_excitation_block(x)
    
    # Сжимаем обратно в 32 фильтра для экономии параметров
    x = Conv1D(32, kernel_size=1, padding='causal', kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    # Residual Connection (добавляем шорткат)
    if int(shortcut.shape[-1]) != 32:
        shortcut = Conv1D(32, 1, padding='valid', kernel_regularizer=regularizers.l2(l2_reg))(shortcut)
        
    x = Add()([shortcut, x])
    x = Activation('gelu')(x)
    return x

def _create_cnn_model(seq_len, n_features, l2_reg):
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    x = GaussianNoise(0.01)(inputs)
    x = LayerNormalization()(x)

    # 1. Bottleneck (Проекция признаков)
    x = Conv1D(32, 1, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    # НОВОЕ: Пространственный дропаут (выключает целые индикаторы на все 6 дней)
    x = SpatialDropout1D(0.15)(x)
    
    # 2. Inception-Residual Блок
    x = inception_residual_block(x, l2_reg)
    
    # 3. Подготовка для Dense слоя
    x = Flatten()(x)
    x = Dropout(0.2)(x) # Чуть усилил финальный дропаут

    # 4. Классификатор
    x = Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(3, activation='softmax', name='out')(x)
    return Model(inputs=inputs, outputs=outputs)

# ==========================================
# ФАБРИКА МОДЕЛЕЙ (ЕДИНАЯ ТОЧКА ВХОДА)
# ==========================================
def create_model(arch, seq_len, n_features, l2_reg):
    if arch == "conv1d+gru":
        return _create_conv1d_gru_model(seq_len, n_features, l2_reg)
    elif arch == "cnn":
        return _create_cnn_model(seq_len, n_features, l2_reg)
    else:
        raise ValueError(f"❌ Неизвестная архитектура: {arch}")

def save_record_model(model, history, acc, loss, train_time, run_id, dataset_name, fold, seq_len, n_features, models_dir, arch):
    # 1. Формируем имя файла с префиксом архитектуры
    model_filename = f"{arch}_loss_{loss:.4f}_acc_{acc*100:.2f}_{run_id}.keras"
    meta_filename = f"{arch}_loss_{loss:.4f}_acc_{acc*100:.2f}_{run_id}.json"
    
    model_path = models_dir / model_filename
    meta_path = models_dir / meta_filename
    
    # 2. Сохраняем модель
    model.save(model_path)
    
    # 3. Сохраняем метаданные
    meta_data = {
        "model_name": model_filename,
        "arch": arch,  # <-- Сохраняем архитектуру в мете
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