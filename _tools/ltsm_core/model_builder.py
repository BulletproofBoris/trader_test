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
    Итерация 5: GRU + Прямой доступ к настоящему (Skip Connection)
    Цель: Сохранить победную логику GRU, но дать классификатору 
    неискаженный слепок рынка в самый последний день (День 6).
    """
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    x = GaussianNoise(0.01)(inputs)
    x = LayerNormalization()(x)

    # 1. ВЕТКА 1: Контекст от GRU (Как в успешной Итерации 3)
    # Выдает сжатый вектор тренда (размер 64)
    gru_context = GRU(
        units=64, 
        return_sequences=False, 
        kernel_regularizer=regularizers.l2(l2_reg),
        name="gru_trend"
    )(x)
    gru_context = Dropout(0.3)(gru_context)

    # 2. ВЕТКА 2: "Прямой Провод" (Shortcut) к Дню 6
    # С помощью Lambda берем только последнюю строку из 6 дней
    # Выдает сырые, неискаженные признаки сегодняшнего дня (размер ~68)
    today_raw = Lambda(lambda s: s[:, -1, :], name="today_shortcut")(x)

    # 3. Слияние: Тренд + Текущая реальность
    # Склеиваем 64 признака от GRU и ~68 сырых признаков Дня 6
    merged = Concatenate(name="fusion")([gru_context, today_raw])

    # 4. Принятие решения
    x = Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(merged)
    x = Dropout(0.3)(x) # Чуть усилили Dropout на горлышке из-за увеличения данных
    
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