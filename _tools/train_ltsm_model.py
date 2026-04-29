import argparse
import json
import sys
import os
import time
import sqlite3
import hashlib
from pathlib import Path
import warnings
import numpy as np

# Прячем системный спам
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed precision включена!")

from tensorflow.keras.layers import (
    Input, Dense, GRU, Bidirectional, Dropout, Attention,
    Add, LayerNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras import regularizers

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Динамическое выделение видеопамяти включено!")
    except RuntimeError as e:
        print(e)


# ==============================================================================
# 🧠 SMART ORCHESTRATOR (Управление БД, Z-Score и Макро-терпение фолда)
# ==============================================================================
class SmartOrchestrator:
    def __init__(self, db_path):
        self.db_path = str(db_path)
        self._create_tables()

    def _execute(self, query, params=(), fetch=False, max_retries=5):
        for attempt in range(max_retries):
            try:
                with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;") 
                    cur = conn.cursor()
                    cur.execute(query, params)
                    if fetch:
                        return cur.fetchall()
                    conn.commit()
                    return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    sleep_time = np.random.uniform(0.1, 0.5)
                    time.sleep(sleep_time)
                else:
                    raise

    def _create_tables(self):
        self._execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                config TEXT,
                fold TEXT,
                hyperparams TEXT,
                val_loss REAL,
                val_acc REAL,
                avg_epoch_time REAL,
                overhead_time REAL,
                total_ttc REAL,
                status TEXT
            )
        """)
        self._execute("""
            CREATE TABLE IF NOT EXISTS folds_meta (
                fold_name TEXT PRIMARY KEY,
                best_loss REAL,
                runs_since_improvement INTEGER,
                status TEXT
            )
        """)

    def register_run_start(self, run_id, config, fold, hyperparams):
        params_json = json.dumps(hyperparams)
        self._execute(
            "INSERT INTO runs (run_id, config, fold, hyperparams, status) VALUES (?, ?, ?, ?, 'TRAINING')",
            (run_id, config, fold, params_json)
        )

    def register_run_end(self, run_id, val_loss, val_acc, avg_epoch_time, overhead_time, total_ttc, status='COMPLETED'):
        self._execute("""
            UPDATE runs 
            SET val_loss=?, val_acc=?, avg_epoch_time=?, overhead_time=?, total_ttc=?, status=? 
            WHERE run_id=?
        """, (val_loss, val_acc, avg_epoch_time, overhead_time, total_ttc, status, run_id))

    def update_fold_meta(self, fold_name, current_loss, max_patience):
        rows = self._execute("SELECT best_loss, runs_since_improvement, status FROM folds_meta WHERE fold_name=?", (fold_name,), fetch=True)
        if not rows:
            self._execute("INSERT INTO folds_meta (fold_name, best_loss, runs_since_improvement, status) VALUES (?, ?, 0, 'OPEN')", 
                          (fold_name, current_loss))
            return
        best_loss, runs_stuck, status = rows[0]
        if status == 'CLOSED': return

        if current_loss < best_loss - 0.0005: 
            self._execute("UPDATE folds_meta SET best_loss=?, runs_since_improvement=0 WHERE fold_name=?", (current_loss, fold_name))
            print(f"🌍 [Оркестратор] Фолд обновлен. Новый глобальный рекорд: {current_loss:.4f}")
        else:
            runs_stuck += 1
            if runs_stuck >= max_patience:
                self._execute("UPDATE folds_meta SET runs_since_improvement=?, status='CLOSED' WHERE fold_name=?", (runs_stuck, fold_name))
                print(f"🌍 [Оркестратор] ФОЛД ЗАКРЫТ. Лимит макро-терпения ({max_patience}) исчерпан.")
            else:
                self._execute("UPDATE folds_meta SET runs_since_improvement=? WHERE fold_name=?", (runs_stuck, fold_name))

    def check_fold_status(self, fold_name):
        rows = self._execute("SELECT status FROM folds_meta WHERE fold_name=?", (fold_name,), fetch=True)
        if rows and rows[0][0] == 'CLOSED':
            return False
        return True

    def should_prune_model(self, fold_name, current_loss, threshold=2.5):
        rows = self._execute("SELECT val_loss FROM runs WHERE fold=? AND status='COMPLETED' AND val_loss IS NOT NULL", (fold_name,), fetch=True)
        losses = [r[0] for r in rows]
        if len(losses) < 5: 
            return False
        mu, sigma = np.mean(losses), np.std(losses)
        if sigma == 0: return False
        
        z_score = (current_loss - mu) / sigma
        # Используем поднятый порог (2.5) для защиты от "холодного старта"
        if z_score > threshold:
            print(f"\n🔪 [Z-Score Pruning] Loss {current_loss:.4f} аномально высок (Z={z_score:.2f}, Порог={threshold}).")
            return True
        return False


# ==============================================================================
# ⏱️ ЭЛАСТИЧНЫЙ ПРОФАЙЛЕР (Динамическое бюджетирование эпох)
# ==============================================================================
class ElasticPatienceProfiler(Callback):
    def __init__(self, orchestrator, fold_name, max_epochs):
        super().__init__()
        self.orchestrator = orchestrator
        self.fold_name = fold_name
        self.max_epochs = max_epochs
        self.epoch_times = []
        self.pruned = False
        
        # Эластичная математика (доли от max_epochs)
        self.micro_patience = max(1, int(0.1 * max_epochs))       # 1/10
        self.macro_patience = max(3.0, float(0.3 * max_epochs))   # 3/10 (Стартовый лимит)
        self.macro_bonus = 0.3 * self.micro_patience              # Бонус за локальный рекорд
        
        self.micro_wait = 0
        self.local_best_loss = np.inf
        
    def on_train_begin(self, logs=None):
        self.run_start_time = time.time()
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_duration)
        
        current_loss = logs.get('val_loss')
        if current_loss is None: return
        
        # 1. Логика Эластичного расширения (Локальные рекорды ВНУТРИ рана)
        if current_loss < self.local_best_loss - 1e-4:
            self.local_best_loss = current_loss
            self.micro_wait = 0
            
            # Зарабатываем эпохи, но не выходим за рамки абсолютного максимума (args.epochs)
            old_macro = self.macro_patience
            self.macro_patience = min(float(self.max_epochs), self.macro_patience + self.macro_bonus)
            if int(self.macro_patience) > int(old_macro):
                print(f"📈 [Бонус] Локальный рекорд! Лимит эпох расширен до {int(self.macro_patience)}.")
        else:
            self.micro_wait += 1

        # 2. Триггер Микро-счетчика (Проверка на мусор)
        if self.micro_wait >= self.micro_patience:
            # Терпение лопнуло. Спрашиваем базу, маргинал ли эта модель
            if self.orchestrator.should_prune_model(self.fold_name, current_loss, threshold=2.5):
                print(f"🛑 [Отсев] Нет улучшений {self.micro_patience} эпох. Z-Score > 2.5. Итерация убита.")
                self.model.stop_training = True
                self.pruned = True
                return
            else:
                # Z-Score в норме (модель терпимая). Прощаем ей заминку.
                self.micro_wait = 0 
                print(f"🔄 [Прощение] Заминка {self.micro_patience} эпох, но Z-Score в норме. Учим дальше.")
                
        # 3. Триггер Макро-счетчика (Early Stopping)
        if (epoch + 1) >= int(self.macro_patience):
            print(f"⏳ [Early Stopping] Обучение остановлено. Достигнут эластичный лимит: {int(self.macro_patience)} эпох.")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.total_ttc = time.time() - self.run_start_time
        clean_epochs = self.epoch_times[1:] if len(self.epoch_times) > 1 else self.epoch_times
        self.avg_epoch_time = float(np.mean(clean_epochs)) if clean_epochs else 0.0
        pure_compute_time = sum(self.epoch_times)
        self.overhead_time = max(0.0, self.total_ttc - pure_compute_time)


# ==============================================================================
# 📉 Smart Backtrack Callback (Снижение LR на локальных ямах)
# ==============================================================================
class SmartBacktrackCallback(Callback):
    def __init__(self, best_weights_path, target_loss, monitor_loss='val_loss', factor=0.5, patience=4, min_lr=1e-6, max_rollbacks=3):
        super(SmartBacktrackCallback, self).__init__()
        self.monitor_loss = monitor_loss
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.max_rollbacks = max_rollbacks
        self.best_weights_path = str(best_weights_path)
        self.target_loss = target_loss 
        
        self.wait = 0
        self.rollback_count = 0
        self.best_loss = np.inf
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor_loss)
        if current_loss is None: return

        if current_loss < self.best_loss - 1e-4:
            self.best_loss = current_loss
            self.best_epoch = epoch + 1
            self.wait = 0
            self.rollback_count = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.rollback_count += 1
                
                if self.rollback_count >= self.max_rollbacks:
                    if self.best_loss >= self.target_loss:
                        print(f"\n🛑 Лимит откатов LR исчерпан. Локальный максимум ({self.best_loss:.4f}) слабее рекорда. СБРОС ИТЕРАЦИИ.")
                    else:
                        print(f"\n🛑 Итерация застряла, но глобальный рекорд УЖЕ ПОБИТ ({self.best_loss:.4f} < {self.target_loss:.4f})! Идем на сохранение.")
                    self.model.stop_training = True
                    return

                if os.path.exists(self.best_weights_path):
                    print(f"\n⚠️ Откат к весам из эпохи #{self.best_epoch} (Откат {self.rollback_count}/{self.max_rollbacks})...")
                    self.model.load_weights(self.best_weights_path)

                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                if old_lr > self.min_lr:
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    self.model.optimizer.learning_rate.assign(new_lr)
                    print(f"📉 Снижаю learning rate до {new_lr:.0e}.")
                    self.wait = 0


# ==============================================================================
# 🔧 ФУНКЦИИ ДАТАСЕТА И МОДЕЛИ
# ==============================================================================
def parse_tfrecord_fn(example, lookback, n_features):
    feature_description = {
        'sequence': tf.io.FixedLenFeature([], tf.string), 
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    sequence = tf.io.parse_tensor(example['sequence'], out_type=tf.float32)
    sequence.set_shape([lookback, n_features])
    label = tf.one_hot(example['target'], depth=3)
    label.set_shape([3])
    return sequence, label

def load_tfrecord_dataset(file_path, batch_size, lookback, n_features, is_training=True):
    dataset = tf.data.TFRecordDataset(str(file_path), num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: parse_tfrecord_fn(x, lookback, n_features), num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.cache().shuffle(10000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def compute_class_weights_fast(tfrecord_path):
    print("⚙️ Расчет идеальных весов классов...")
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    class_counts = {0: 0, 1: 0, 2: 0}
    feature_description = {'target': tf.io.FixedLenFeature([], tf.int64)}
    
    for raw_record in dataset:
        parsed = tf.io.parse_single_example(raw_record, feature_description)
        class_counts[int(parsed['target'].numpy())] += 1
        
    total = sum(class_counts.values())
    weights = {c: total / (3.0 * max(1, count)) for c, count in class_counts.items()}
    print(f"   Баланс: SL(0)={class_counts[0]}, Hold(1)={class_counts[1]}, TP(2)={class_counts[2]}")
    print(f"   Веса:   SL(0)={weights[0]:.2f}, Hold(1)={weights[1]:.2f}, TP(2)={weights[2]:.2f}")
    return weights

def create_model(seq_len, n_features, l2_reg=1e-5):
    inputs = Input(shape=(seq_len, n_features), name="input_layer")
    x = Dense(64, activation='linear', name='fp16_projection')(inputs)
    x = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg)))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=regularizers.l2(l2_reg)))(x)
    res_x = Dropout(0.2)(x)
    
    attn_out = Attention()([res_x, res_x])
    x = Add()([res_x, attn_out])
    x = LayerNormalization()(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    
    x = Dense(64, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Activation('gelu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(3, activation='softmax', name='out', dtype='float32')(x)
    return Model(inputs=inputs, outputs=outputs)

def save_record_model(model, history, acc, loss, train_time, run_id, args, seq_len, n_features, models_dir):
    timestamp = int(time.time())
    model_filename = f"trading_bot_best_acc_{acc*100:.2f}_{run_id}.keras"
    model_filepath = models_dir / model_filename
    model.save(model_filepath)
    
    meta_filepath = models_dir / f"trading_bot_best_acc_{acc*100:.2f}_{run_id}.json"
    clean_history = {k: [float(val) for val in v] for k, v in history.history.items()} if history else {}
    
    dataset_name = Path(args.dataset_dir).name
    parts = dataset_name.split('_')
    horizon = int(parts[-1]) if len(parts) >= 2 and parts[-1].isdigit() else "unknown"
    
    report = {
        "model_name": model_filename,
        "run_id": str(run_id),
        "dataset": dataset_name,
        "fold": args.fold,
        "config": {"lookback": seq_len, "horizon": horizon, "features_count": n_features, "architecture": "BiGRU + Attention"},
        "metrics": {"val_loss": float(loss), "val_acc": float(acc * 100)},
        "training_stats": {
            "timestamp": timestamp,
            "training_time_seconds": float(train_time),
            "hyperparameters": {"batch_size": args.batch_size, "max_epochs": args.epochs, "start_lr": args.lr, "l2_reg": args.l2_reg},
            "training_history": clean_history
        }
    }
    with open(meta_filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)


# ==============================================================================
# 🚀 MAIN LOOP
# ==============================================================================
def main(args):
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASET_DIR = BASE_DIR / args.dataset_dir
    FOLD_DIR = DATASET_DIR / args.fold
    
    TFRECORDS_DIR = FOLD_DIR / "data"
    MODELS_DIR = FOLD_DIR / "models"
    ARTIFACTS_DIR = FOLD_DIR / "artifacts"
    
    db_path = FOLD_DIR / "trading_factory.db"
    orchestrator = SmartOrchestrator(db_path)
    
    global_best_loss = float('inf')
    
    if MODELS_DIR.exists() and any(MODELS_DIR.glob("*.keras")):
        if args.force:
            print(f"⚠️ Флаг --force: Очищаем папку моделей фолда [{args.fold}]...")
            for f in MODELS_DIR.glob("*"): 
                f.unlink()
        elif args.append:
            print(f"➕ Флаг --append: Ищем исторический рекорд в прошлых моделях...")
            for f in MODELS_DIR.glob("*.json"):
                try:
                    with open(f, 'r') as jf:
                        m_data = json.load(jf)
                        m_loss = m_data.get("metrics", {}).get("val_loss", float('inf'))
                        if m_loss < global_best_loss:
                            global_best_loss = m_loss
                except Exception:
                    pass
            print(f"🛡️ Базовый рекорд для побития (Loss): {global_best_loss if global_best_loss != float('inf') else 'Нет данных'}")
        else:
            print(f"✅ В фолде [{args.fold}] уже есть обученные модели.")
            return
                
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    dataset_meta_path = DATASET_DIR / "metadata.json"
    with open(dataset_meta_path, 'r', encoding='utf-8') as f:
        seq_len = json.load(f)["parameters"]["lookback"]

    features_json = ARTIFACTS_DIR / "features_selected.json"
    with open(features_json, 'r', encoding='utf-8') as f:
        n_features = len(json.load(f).get("feature_order", []))
            
    print(f"🚀 Старт обучения. Фолд: [{args.fold}] | Форма: [{seq_len}, {n_features}]")
    
    train_record_path = TFRECORDS_DIR / "train" / "data.tfrecord"
    val_record_path = TFRECORDS_DIR / "val" / "data.tfrecord"
    
    class_weights_dict = compute_class_weights_fast(train_record_path)
    train_dataset = load_tfrecord_dataset(train_record_path, args.batch_size, seq_len, n_features, is_training=True)
    val_dataset = load_tfrecord_dataset(val_record_path, args.batch_size, seq_len, n_features, is_training=False)

    max_patience_runs = max(5, int(args.runs * 0.20))

    for run in range(1, args.runs + 1):
        if not orchestrator.check_fold_status(args.fold):
            print(f"\n{'='*50}\n🛑 ГЛОБАЛЬНАЯ ОСТАНОВКА: Фолд {args.fold} закрыт Оркестратором.\nЛимит макро-терпения исчерпан. Дальнейшее обучение бессмысленно.\n{'='*50}")
            break

        print(f"\n{'-'*50}\n🔄 ИТЕРАЦИЯ {run}/{args.runs} (Цель Loss < {global_best_loss:.4f})\n{'-'*50}")
        
        run_id = f"run_{hashlib.md5(f'{time.time()}_{np.random.randint(1000)}'.encode()).hexdigest()[:8]}"
        hyperparams = {"lr": args.lr, "batch": args.batch_size, "l2": args.l2_reg}
        orchestrator.register_run_start(run_id, Path(args.dataset_dir).name, args.fold, hyperparams)
        
        tf.keras.backend.clear_session()
        model = create_model(seq_len, n_features, args.l2_reg)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )

        temp_weights_path = MODELS_DIR / f"temp_best_{run_id}.weights.h5"
        
        # Эластичный профайлер (заменяет классический EarlyStopping)
        profiler = ElasticPatienceProfiler(orchestrator, args.fold, args.epochs)
        
        callbacks = [
            ModelCheckpoint(filepath=temp_weights_path, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True, verbose=0),
            SmartBacktrackCallback(best_weights_path=temp_weights_path, target_loss=global_best_loss, monitor_loss='val_loss', patience=4),
            profiler
        ]

        try:
            history = model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=callbacks, class_weight=class_weights_dict, verbose=2)
        except KeyboardInterrupt:
            print("\n⚠️ Multi-Run прерван пользователем.")
            break

        # Восстанавливаем веса лучшей эпохи перед оценкой
        if os.path.exists(temp_weights_path):
            model.load_weights(temp_weights_path)
            os.remove(temp_weights_path)
            
        loss, acc = model.evaluate(val_dataset, verbose=0)
        
        status = 'PRUNED' if profiler.pruned else 'COMPLETED'
        orchestrator.register_run_end(
            run_id=run_id, val_loss=loss, val_acc=acc,
            avg_epoch_time=profiler.avg_epoch_time, overhead_time=profiler.overhead_time,
            total_ttc=profiler.total_ttc, status=status
        )
        
        print(f"\n📊 МЕТРИКИ ВРЕМЕНИ (run_id: {run_id}):")
        print(f"   Чистая эпоха: {profiler.avg_epoch_time:.2f} сек")
        print(f"   Накладные расходы: {profiler.overhead_time:.2f} сек")
        print(f"   Общее время (TTC): {profiler.total_ttc:.2f} сек")
        
        if profiler.pruned:
            print(f"⏭️ Пропуск сохранения. Модель была отсеяна Оркестратором.")
            continue

        print(f"\n🎯 Итог итерации {run}: Val Loss = {loss:.4f} | Val Acc = {acc*100:.2f}%")
        
        if loss < global_best_loss:
            print(f"🏆 НОВЫЙ РЕКОРД по Loss! {global_best_loss:.4f} -> {loss:.4f}")
            global_best_loss = loss
            save_record_model(model, history, acc, loss, profiler.total_ttc, run_id, args, seq_len, n_features, MODELS_DIR)
            print(f"💾 Модель успешно сохранена!")
            
        orchestrator.update_fold_meta(args.fold, loss, max_patience_runs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/processed/2000_2026_1d", help="Папка датасета")
    parser.add_argument("--fold", type=str, default="fold_2010", help="Имя фолда для обучения")
    parser.add_argument("--runs", type=int, default=10, help="Количество циклов инициализации весов")
    parser.add_argument("--batch_size", type=int, default=512, help="Размер батча")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох")
    parser.add_argument("--lr", type=float, default=1e-3, help="Стартовый Learning Rate")
    parser.add_argument("--l2_reg", type=float, default=1e-5, help="L2 регуляризация")
    parser.add_argument("--force", action="store_true", help="Принудительное обучение с удалением старых моделей")
    parser.add_argument("--append", action="store_true", help="Добавить новые модели, не удаляя старые")
    args = parser.parse_args()
    main(args)