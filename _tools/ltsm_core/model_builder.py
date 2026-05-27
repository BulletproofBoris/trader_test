import json
from pathlib import Path
import tensorflow as tf
import math
from tensorflow.keras.layers import Input, LayerNormalization, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, LSTM, Dropout, Dense, Flatten, Activation, Dot, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def create_model(seq_len, n_features, l2_reg):
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    x = LayerNormalization()(inputs)

    # ==========================================
    # ⚡ ВЕТКА 1: АДАПТИВНЫЙ ИМПУЛЬС (CNN)
    # ==========================================
    # Базовое количество фильтров (Адаптивно: растет от фичей, потолок 128)
    base_filters = min(128, max(32, int(n_features * 0.5)))
    
    # Динамические ядра (Inception-style): зависят от длины окна
    p = 0.05 * seq_len
    kernel_1 = max(1, math.ceil(p * 1))
    kernel_2 = max(2, math.ceil(p * 2))
    kernel_3 = max(3, math.ceil(p * 3))
    kernel_4 = max(4, math.ceil(p * 4))
    
    # Слой 1: Параллельные ветки сверток (Causal padding предотвращает заглядывание вперед)
    conv_1 = Conv1D(filters=base_filters, kernel_size=kernel_1, padding='causal', activation='relu')(x)
    c1_last = Lambda(lambda s: s[:, -1, :], name="cnn_last_1")(conv_1)

    conv_2 = Conv1D(filters=base_filters, kernel_size=kernel_2, padding='causal', activation='relu')(x)
    c2_last = Lambda(lambda s: s[:, -1, :], name="cnn_last_2")(conv_2)


    # Слой 2: Углубление параллельных веток
    conv_3 = Conv1D(filters=base_filters, kernel_size=kernel_3, padding='causal', activation='relu')(conv_1)
    c3_last = Lambda(lambda s: s[:, -1, :], name="cnn_last_3")(conv_3)

    conv_4 = Conv1D(filters=base_filters, kernel_size=kernel_4, padding='causal', activation='relu')(conv_2)
    c4_last = Lambda(lambda s: s[:, -1, :], name="cnn_last_4")(conv_4)

    # Конкатенация выходов CNN (Размер: base_filters * 2)
    conv_out = Concatenate(name="cnn_concat")([c1_last, c2_last]) 

    # ==========================================
    # 🧠 ВЕТКА 2: ГЛУБОКИЙ КОНТЕКСТ (LSTM + Attention)
    # ==========================================
    # АДАПТИВНАЯ ПАМЯТЬ: Зависит от количества фичей, но не превышает 128 для защиты от переобучения
    lstm_base = min(128, max(16, int(seq_len * 2)))
    lstm_base = 16

    # Уровень 1: Грубое впитывание истории
    lstm = LSTM(lstm_base, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(x)
    lstm = Dropout(0.05)(lstm)

    # Уровень 2: Тонкое сжатие (делим память на 2, создаем воронку)
    lstm_out = LSTM(lstm_base // 2, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg))(lstm)
    lstm_out = Dropout(0.05)(lstm_out)

    # Механизм внимания (Оценивает важность каждого шага из истории)
    attention_scores = Dense(1, activation='linear')(lstm_out)
    
    # СОВЕТ: Если на 90_15 все равно будет плохо, раскомментируй строку ниже (Temperature Scaling)
    # attention_scores = Lambda(lambda s: s / 0.5)(attention_scores)
    
    attention_scores = Flatten()(attention_scores)
    attention_weights = Activation('softmax', name='attention_weights')(attention_scores)
    
    # Умножаем веса на выходы LSTM, получая один вектор контекста (Размер: lstm_base // 2)
    lstm_att = Dot(axes=1, name="lstm_context")([attention_weights, lstm_out])

    # ==========================================
    # 🧬 СЛИЯНИЕ И ФИНАЛ
    # ==========================================
    # Склеиваем ветки: CNN (Локальные импульсы) + LSTM (Исторический контекст)
    merged = Concatenate(name="fusion_concat")([conv_out, lstm_att])
    merged = LayerNormalization()(merged)
    
    # АДАПТИВНОЕ ГОРЛЫШКО (Dense): 
    # Вычисляем размер получившегося вектора и сжимаем его в 2 раза
    merged_dim = (base_filters * 2) + (lstm_base // 2)
    dense_units = min(128, max(4, int(merged_dim * 0.5)))
    
    x = Dense(dense_units, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(merged)
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