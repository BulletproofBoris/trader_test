import json
from pathlib import Path
import math
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization, 
                                     Conv1D, GaussianNoise, GRU, Multiply,
                                     Add, Activation, Flatten)
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
# 2. АРХИТЕКТУРА БЕЗ РЕКУРСИИ (cnn) - Минималистичная V1
# ==========================================
def residual_conv_block(x, filters, kernel_size, l2_reg):
    shortcut = x
    
    # Строго causal паддинг для временных рядов!
    x = Conv1D(filters, kernel_size, padding='causal', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = LayerNormalization()(x)
    x = Activation('gelu')(x)
    x = Dropout(0.1)(x)
    
    x = Conv1D(filters, kernel_size, padding='causal', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = LayerNormalization()(x)
    
    # Выравнивание размерностей (с padding='valid', т.к. ядро 1 не съедает длину)
    if int(shortcut.shape[-1]) != filters:
        shortcut = Conv1D(filters, 1, padding='valid', kernel_regularizer=regularizers.l2(l2_reg))(shortcut)
        
    x = Add()([shortcut, x])
    x = Activation('gelu')(x)
    return x

def _create_cnn_model(seq_len, n_features, l2_reg):
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    x = GaussianNoise(0.01)(inputs)
    x = LayerNormalization()(x)

    # 1. Bottleneck (Проекция признаков + Регуляризация)
    x = Conv1D(16, 1, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    # 2. Простая остаточная свертка
    x = residual_conv_block(x, filters=16, kernel_size=4, l2_reg=l2_reg)
    
    # 3. Вытягиваем в вектор (Без потери времени!)
    x = Flatten()(x)
    x = Dropout(0.15)(x)

    # 4. Классификатор
    x = Dense(32, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(0.15)(x)
    
    outputs = Dense(3, activation='softmax', name='out')(x)
    return Model(inputs=inputs, outputs=outputs)

# ==========================================
# 3. АРХИТЕКТУРА ТАБЛИЧНАЯ (mlp) - Глобальный взгляд
# ==========================================
def _create_mlp_model(seq_len, n_features, l2_reg):
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    x = GaussianNoise(0.01)(inputs)
    x = LayerNormalization()(x)

    # Мгновенно уничтожаем понятие времени: (6, 68) -> (408,)
    x = Flatten()(x)

    # 1. Проекция в широкое пространство (Даем сети возможность дышать)
    # Обрати внимание: активации внутри Dense НЕТ! Мы делаем это вручную.
    x = Dense(512, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = LayerNormalization()(x)
    x = Activation('gelu')(x)
    x = Dropout(0.15)(x)

    # 2. Residual Блок (MLP-ResNet) - Главный секрет для табличных данных
    shortcut = x
    x = Dense(512, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = LayerNormalization()(x)
    x = Activation('gelu')(x)
    x = Dropout(0.15)(x)
    
    # Складываем вход и выход блока
    x = Add()([shortcut, x])

    # 3. Мягкое сужение перед классификатором
    x = Dense(64, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = LayerNormalization()(x)
    x = Activation('gelu')(x)
    x = Dropout(0.15)(x) # Снизили дропаут, чтобы не убить сигнал
    
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
    elif arch == "mlp":
        return _create_mlp_model(seq_len, n_features, l2_reg)
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