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
import pandas as pd
from scipy.optimize import curve_fit

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
# 🧮 МАТЕМАТИЧЕСКИЙ МАКРО-АНАЛИЗАТОР (HPO Scaling Laws)
# ==============================================================================
class MathTrendAnalyzer:
    @staticmethod
    def decay_model(x, a, b, c):
        return a * np.exp(-b * x) + c

    @staticmethod
    def calculate_macro_trend(losses_array, min_margin_pct=0.001, max_margin_pct=0.005):
        if len(losses_array) < 15: 
            return None, None, None
        
        df = pd.DataFrame({'loss': losses_array})
        
        q1 = df['loss'].quantile(0.25)
        q3 = df['loss'].quantile(0.75)
        iqr = q3 - q1
        smart_max = q3 + 1.5 * iqr
        p85 = df['loss'].quantile(0.85)
        valid_max = min(smart_max, p85)
        
        df['best_so_far'] = df['loss'].cummin()
        x_data = np.arange(1, len(df) + 1)
        y_data = df['best_so_far'].values
        
        valid_mask = (y_data >= 0.0) & (y_data <= valid_max)
        x_fit = x_data[valid_mask]
        y_fit = y_data[valid_mask]
        
        if len(x_fit) < 5: 
            return None, None, None
            
        amplitude = y_fit[0] - y_fit[-1]
        if amplitude <= 0: amplitude = 0.01
            
        y_target = y_fit[-1]
        initial_guess = [amplitude, 0.05, max(0, y_target - 0.01)]
        lower_bounds = [0, 0, 0]
        upper_bounds = [np.inf, np.inf, max(1e-5, y_target)]
        
        try:
            popt, pcov = curve_fit(
                MathTrendAnalyzer.decay_model, 
                x_fit, 
                y_fit, 
                p0=initial_guess, 
                bounds=(lower_bounds, upper_bounds), 
                maxfev=10000
            )
            a, b, c = popt
            
            variance_c = pcov[2][2] if not np.isinf(pcov[2][2]) else 0.0
            std_c = np.sqrt(variance_c)
            
            statistical_margin = std_c * 2.0 
            abs_min_margin = c * min_margin_pct 
            abs_max_margin = c * max_margin_pct 
            
            margin = np.clip(statistical_margin, abs_min_margin, abs_max_margin)
            
            target_loss = c + margin
            if target_loss >= y_fit[0]:
                runs_to_margin = 0 
            else:
                runs_to_margin = -np.log((target_loss - c) / a) / b
                runs_to_margin = int(np.ceil(runs_to_margin))
                
            return c, margin, runs_to_margin
            
        except Exception:
            return None, None, None


# ==============================================================================
# 🧠 SMART ORCHESTRATOR (Swarm Control)
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
                    time.sleep(np.random.uniform(0.1, 0.5))
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
        self._execute("CREATE TABLE IF NOT EXISTS folds_meta (fold_name TEXT PRIMARY KEY, best_loss REAL)")
        self._execute("""
            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY,
                fold TEXT,
                remaining_runs INTEGER,
                last_seen REAL
            )
        """)

    def update_worker_heartbeat(self, worker_id, fold_name, remaining_runs):
        """Регистрирует воркер и его оставшийся бюджет, удаляет зависшие процессы (> 10 мин)"""
        current_time = time.time()
        self._execute(
            "INSERT OR REPLACE INTO workers (worker_id, fold, remaining_runs, last_seen) VALUES (?, ?, ?, ?)",
            (worker_id, fold_name, remaining_runs, current_time)
        )
        self._execute("DELETE FROM workers WHERE last_seen < ?", (current_time - 600,))

    def remove_worker(self, worker_id):
        """Штатный выход воркера из пула"""
        self._execute("DELETE FROM workers WHERE worker_id=?", (worker_id,))

    def sync_with_filesystem(self, models_dir, fold_name, dataset_name):
        total_files = 0
        added_count = 0
        best_synced_loss = float('inf')

        for meta_file in Path(models_dir).glob("*.json"):
            total_files += 1
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                run_id = data.get("run_id")
                loss = data.get("metrics", {}).get("val_loss")
                acc = data.get("metrics", {}).get("val_acc", 0) / 100.0 
                ttc = data.get("training_stats", {}).get("training_time_seconds", 0.0)

                if not run_id or loss is None:
                    continue

                exists = self._execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,), fetch=True)
                if not exists:
                    self._execute(
                        """INSERT INTO runs 
                           (run_id, config, fold, hyperparams, val_loss, val_acc, avg_epoch_time, overhead_time, total_ttc, status) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'COMPLETED')""",
                        (run_id, dataset_name, fold_name, "{}", loss, acc, 0.0, 0.0, ttc)
                    )
                    added_count += 1
                    best_synced_loss = min(best_synced_loss, loss)

            except Exception as e:
                print(f"⚠️ Ошибка чтения файла {meta_file.name}: {e}")

        if total_files > 0:
            if added_count > 0:
                print(f"🔄 [Синхронизация] Проверено файлов: {total_files}. Добавлено {added_count} записей.")
                rows = self._execute("SELECT best_loss FROM folds_meta WHERE fold_name=?", (fold_name,), fetch=True)
                current_db_best = rows[0][0] if rows else float('inf')

                if best_synced_loss < current_db_best:
                    self._execute("INSERT OR REPLACE INTO folds_meta (fold_name, best_loss) VALUES (?, ?)", (fold_name, best_synced_loss))
                    print(f"🌍 [Оркестратор] Глобальный рекорд БД обновлен из найденных файлов: {best_synced_loss:.4f}")
            else:
                print(f"✅ [Синхронизация] Проверено файлов: {total_files}. База актуальна.")

    def get_history(self, fold_name):
        rows = self._execute(
            "SELECT val_loss FROM runs WHERE fold=? AND status='COMPLETED' AND val_loss IS NOT NULL ORDER BY rowid ASC",
            (fold_name,), fetch=True
        )
        return [r[0] for r in rows]

    def evaluate_potential(self, fold_name, worker_id, remaining_runs, current_run_index):
        """
        Динамическая проверка с учетом мощности ВСЕГО пула.
        """
        # 1. Отправляем пульс и регистрируем свой остаток
        self.update_worker_heartbeat(worker_id, fold_name, remaining_runs)

        # 2. Получаем глобальный рекорд
        meta_rows = self._execute("SELECT best_loss FROM folds_meta WHERE fold_name=?", (fold_name,), fetch=True)
        global_best = meta_rows[0][0] if meta_rows else float('inf')

        # 3. Вычисляем суммарный бюджет пула
        res = self._execute("SELECT SUM(remaining_runs), COUNT(worker_id) FROM workers WHERE fold=?", (fold_name,), fetch=True)
        pool_budget = res[0][0] if res and res[0][0] else 0
        active_workers = res[0][1] if res else 0

        # 🐝 ИГНОРИРОВАНИЕ ПРОВЕРОК НА ПЕРВОЙ ИТЕРАЦИИ (ЗАЩИТА ОТ ГОНКИ / RACE CONDITION)
        if current_run_index == 1:
            return True, f"Прогрев воркера (Итерация 1). В пуле: {active_workers} процесс(ов). Ожидаем остальных...", global_best

        # 4. Анализ тренда (со 2-й итерации)
        losses = self.get_history(fold_name)

        if len(losses) < 15:
            return True, f"Сбор статистики пулом ({len(losses)}/15)... Воркеров: {active_workers}, Резерв пула: {pool_budget}", global_best

        c, margin, required_runs = MathTrendAnalyzer.calculate_macro_trend(losses)
        
        if c is None:
            return True, f"Анализ недоступен. Продолжаем. Воркеров: {active_workers}, Резерв пула: {pool_budget}", global_best

        neighborhood_top = c + margin
        
        if global_best <= neighborhood_top:
            return False, f"Достигнут предел. Рекорд ({global_best:.4f}) вошел в стат. окрестность асимптоты ({c:.4f} + {margin:.4f})", global_best
        
        if required_runs > pool_budget:
            return False, f"Нехватка мощности пула. Нужно ~{required_runs} ранов, а у {active_workers} активных процессов осталось {pool_budget}.", global_best

        return True, f"Пул справится (Воркеров: {active_workers}, Резерв пула: {pool_budget}). Цель: ~{required_runs} ранов до {c:.4f}", global_best

    def register_run_start(self, run_id, config, fold, hyperparams):
        params_json = json.dumps(hyperparams)
        self._execute(
            "INSERT INTO runs (run_id, config, fold, hyperparams, status) VALUES (?, ?, ?, ?, 'TRAINING')",
            (run_id, config, fold, params_json)
        )

    def register_run_end(self, run_id, fold_name, val_loss, val_acc, avg_epoch_time, overhead_time, total_ttc, status='COMPLETED'):
        self._execute("""
            UPDATE runs 
            SET val_loss=?, val_acc=?, avg_epoch_time=?, overhead_time=?, total_ttc=?, status=? 
            WHERE run_id=?
        """, (val_loss, val_acc, avg_epoch_time, overhead_time, total_ttc, status, run_id))
        
        if val_loss is not None:
            rows = self._execute("SELECT best_loss FROM folds_meta WHERE fold_name=?", (fold_name,), fetch=True)
            if not rows or val_loss < rows[0][0]:
                self._execute("INSERT OR REPLACE INTO folds_meta (fold_name, best_loss) VALUES (?, ?)", (fold_name, val_loss))
                print(f"🌍 [Оркестратор] Фолд обновлен. Новый глобальный рекорд БД: {val_loss:.4f}")

    def should_prune_model(self, fold_name, current_loss, threshold=2.0):
        losses = self.get_history(fold_name)
        if len(losses) < 5: 
            return False
        mu, sigma = np.mean(losses), np.std(losses)
        if sigma == 0: return False
        
        z_score = (current_loss - mu) / sigma
        if z_score > threshold:
            print(f"\n🔪 [Z-Score Pruning] Loss {current_loss:.4f} аномально высок (Z={z_score:.2f}).")
            return True
        return False


# ==============================================================================
# ⏱️ ЭЛАСТИЧНЫЙ ПРОФАЙЛЕР (Микро-уровень)
# ==============================================================================
class ElasticPatienceProfiler(Callback):
    def __init__(self, orchestrator, fold_name, max_epochs, bonus_ratio=0.1, min_delta=0.001):
        super().__init__()
        self.orchestrator = orchestrator
        self.fold_name = fold_name
        self.max_epochs = max_epochs
        self.epoch_times = []
        self.pruned = False
        
        self.micro_patience = max(1, int(0.1 * max_epochs))
        self.macro_patience = max(3.0, float(0.3 * max_epochs))
        self.macro_bonus = bonus_ratio * self.micro_patience 
        self.min_delta = min_delta 
        
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
        
        if current_loss < self.local_best_loss - 1e-4:
            self.local_best_loss = current_loss
            self.micro_wait = 0
            self.macro_patience = min(float(self.max_epochs), self.macro_patience + self.macro_bonus)
        else:
            self.micro_wait += 1

        if self.micro_wait >= self.micro_patience:
            if self.orchestrator.should_prune_model(self.fold_name, current_loss, threshold=2.0):
                print(f"🛑 [Отсев] Нет улучшений {self.micro_patience} эпох. Z-Score > 2.0. Итерация убита.")
                self.model.stop_training = True
                self.pruned = True
                return
            else:
                self.micro_wait = 0 
                
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
# 📉 Smart Backtrack Callback 
# ==============================================================================
class SmartBacktrackCallback(Callback):
    def __init__(self, best_weights_path, monitor_loss='val_loss', factor=0.5, patience=4, min_lr=1e-6, max_rollbacks=3):
        super(SmartBacktrackCallback, self).__init__()
        self.monitor_loss = monitor_loss
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.max_rollbacks = max_rollbacks
        self.best_weights_path = str(best_weights_path)
        
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
                    print(f"\n🛑 Лимит откатов LR исчерпан.")
                    self.model.stop_training = True
                    return

                if os.path.exists(self.best_weights_path):
                    self.model.load_weights(self.best_weights_path)

                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                if old_lr > self.min_lr:
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    self.model.optimizer.learning_rate.assign(new_lr)
                    self.wait = 0


# ==============================================================================
# 🔧 ФУНКЦИИ ДАТАСЕТА И МОДЕЛИ
# ==============================================================================
def parse_tfrecord_fn(example, lookback, n_features):
    feature_description = {'sequence': tf.io.FixedLenFeature([], tf.string), 'target': tf.io.FixedLenFeature([], tf.int64)}
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
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    class_counts = {0: 0, 1: 0, 2: 0}
    feature_description = {'target': tf.io.FixedLenFeature([], tf.int64)}
    for raw_record in dataset:
        parsed = tf.io.parse_single_example(raw_record, feature_description)
        class_counts[int(parsed['target'].numpy())] += 1
    total = sum(class_counts.values())
    weights = {c: total / (3.0 * max(1, count)) for c, count in class_counts.items()}
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
    model_filename = f"trading_bot_best_acc_{acc*100:.2f}_{run_id}.keras"
    model.save(models_dir / model_filename)
    
    meta_filepath = models_dir / f"trading_bot_best_acc_{acc*100:.2f}_{run_id}.json"
    clean_history = {k: [float(val) for val in v] for k, v in history.history.items()} if history else {}
    
    report = {
        "model_name": model_filename,
        "run_id": str(run_id),
        "dataset": Path(args.dataset_dir).name,
        "fold": args.fold,
        "metrics": {"val_loss": float(loss), "val_acc": float(acc * 100)},
        "training_stats": {"training_time_seconds": float(train_time)}
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
    
    if MODELS_DIR.exists() and any(MODELS_DIR.glob("*.keras")):
        if args.force:
            for f in MODELS_DIR.glob("*"): f.unlink()
        elif not args.append:
            print(f"✅ В фолде [{args.fold}] уже есть обученные модели (Используйте --append или --force).")
            return
                
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.append and MODELS_DIR.exists():
        dataset_name = Path(args.dataset_dir).name
        orchestrator.sync_with_filesystem(MODELS_DIR, args.fold, dataset_name)
    
    with open(DATASET_DIR / "metadata.json", 'r', encoding='utf-8') as f:
        seq_len = json.load(f)["parameters"]["lookback"]

    with open(ARTIFACTS_DIR / "features_selected.json", 'r', encoding='utf-8') as f:
        n_features = len(json.load(f).get("feature_order", []))
            
    train_record_path = TFRECORDS_DIR / "train" / "data.tfrecord"
    val_record_path = TFRECORDS_DIR / "val" / "data.tfrecord"
    
    class_weights_dict = compute_class_weights_fast(train_record_path)
    train_dataset = load_tfrecord_dataset(train_record_path, args.batch_size, seq_len, n_features, is_training=True)
    val_dataset = load_tfrecord_dataset(val_record_path, args.batch_size, seq_len, n_features, is_training=False)

    # 🐝 Генерируем уникальный ID для этого конкретного процесса
    worker_id = f"worker_{os.getpid()}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}"
    print(f"🚀 Старт обучения. Фолд: [{args.fold}] | Воркер зарегистрирован: {worker_id}")

    try:
        for run in range(1, args.runs + 1):
            
            # Считаем, сколько ранов осталось именно у этого воркера
            remaining_runs_for_this_worker = args.runs - run + 1
            
            # 🐝 ПЕРЕДАЕМ ТЕКУЩИЙ ИНДЕКС ИТЕРАЦИИ ДЛЯ БЛОКИРОВКИ ПРОВЕРОК НА СТАРТЕ
            can_continue, reason, global_best_loss = orchestrator.evaluate_potential(
                args.fold, worker_id, remaining_runs_for_this_worker, current_run_index=run
            )
            
            if not can_continue:
                print(f"\n{'='*60}\n🛑 ГЛОБАЛЬНАЯ ОСТАНОВКА ФОЛДА: {reason}\n{'='*60}")
                break

            print(f"\n{'-'*60}\n🔄 ИТЕРАЦИЯ {run}/{args.runs} (Цель Loss < {global_best_loss:.4f})")
            print(f"📈 Статус макро-тренда: {reason}\n{'-'*60}")
            
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
            profiler = ElasticPatienceProfiler(orchestrator, args.fold, args.epochs, args.bonus_ratio, args.min_delta)
            
            callbacks = [
                ModelCheckpoint(filepath=temp_weights_path, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True, verbose=0),
                SmartBacktrackCallback(best_weights_path=temp_weights_path, monitor_loss='val_loss', patience=4),
                profiler
            ]

            try:
                history = model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=callbacks, class_weight=class_weights_dict, verbose=2)
            except KeyboardInterrupt:
                print("\n⚠️ Multi-Run прерван пользователем.")
                break

            if os.path.exists(temp_weights_path):
                model.load_weights(temp_weights_path)
                os.remove(temp_weights_path)
                
            loss, acc = model.evaluate(val_dataset, verbose=0)
            
            status = 'PRUNED' if profiler.pruned else 'COMPLETED'
            orchestrator.register_run_end(
                run_id=run_id, fold_name=args.fold, val_loss=loss, val_acc=acc,
                avg_epoch_time=profiler.avg_epoch_time, overhead_time=profiler.overhead_time,
                total_ttc=profiler.total_ttc, status=status
            )
            
            if profiler.pruned:
                continue

            print(f"\n🎯 Итог итерации {run}: Val Loss = {loss:.4f} | Val Acc = {acc*100:.2f}%")
            
            if loss < global_best_loss:
                save_record_model(model, history, acc, loss, profiler.total_ttc, run_id, args, seq_len, n_features, MODELS_DIR)
                print(f"🏆 Модель-рекордсмен успешно сохранена!")

    finally:
        # 🐝 Корректно выходим из пула по завершении работы или при ошибке
        orchestrator.remove_worker(worker_id)
        print(f"👋 Воркер {worker_id} освободил мощности и покинул пул.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/processed/2000_2026_1d", help="Папка датасета")
    parser.add_argument("--fold", type=str, default="fold_2010", help="Имя фолда для обучения")
    parser.add_argument("--runs", type=int, default=100, help="Общий бюджет (Количество ранов)")
    parser.add_argument("--batch_size", type=int, default=512, help="Размер батча")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох")
    parser.add_argument("--lr", type=float, default=1e-3, help="Стартовый Learning Rate")
    parser.add_argument("--l2_reg", type=float, default=1e-5, help="L2 регуляризация")
    parser.add_argument("--force", action="store_true", help="Принудительное обучение с удалением старых моделей")
    parser.add_argument("--append", action="store_true", help="Добавить новые модели, не удаляя старые")
    parser.add_argument("--bonus_ratio", type=float, default=0.1, help="Доля от микро-лимита, добавляемая за рекорд")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Минимальное улучшение Loss для получения бонуса")
    args = parser.parse_args()
    main(args)