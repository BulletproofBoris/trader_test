import argparse
import gc
import os
import time
import sys
import hashlib
from pathlib import Path
import warnings
import numpy as np
import json
import ctypes
import multiprocessing

# Жестко прописываем путь к корню инструментов
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Прячем системный спам
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Включаем асинхронное выделение памяти (Снижает потребление VRAM до 50%!)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf

# 1. Включаем кэширование компиляции XLA на диск
os.environ['TF_XLA_FLAGS'] = "--tf_xla_persistent_cache_directory=./xla_cache"

# 2. Жестко ограничиваем потоки для каждого воркеров!
num_cores = multiprocessing.cpu_count()
workers_count = 16 # Твой MAX_WORKERS из bash-скрипта
threads_per_worker = max(2, num_cores // workers_count)

tf.config.threading.set_inter_op_parallelism_threads(threads_per_worker)
tf.config.threading.set_intra_op_parallelism_threads(threads_per_worker)

print(f"⚙️ CPU-Квота: выделено {threads_per_worker} потоков на этот процесс.")

# Форсируем включение аппаратного TF32 для тензорных ядер (на всякий случай)
tf.config.experimental.enable_tensor_float_32_execution(True)
print("✅ Аппаратное ускорение TF32 включено!")

# Включаем Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        VRAM_LIMIT_MB = 4000 
        
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=VRAM_LIMIT_MB)]
            )
        print(f"✅ Жесткий квотированный лимит VRAM: {VRAM_LIMIT_MB} МБ на процесс!")
    except RuntimeError as e: 
        print(e)

# --- Импорты из ядра ---
from ltsm_core.orchestrator import SmartOrchestrator
from ltsm_core.data_loader import compute_class_weights_fast, load_tfrecord_dataset, count_tfrecord_samples
from ltsm_core.model_builder import create_model, save_record_model
# ИСПРАВЛЕНО: Добавлен импорт FullTrajectoryTracker из вашего файла callbacks
from ltsm_core.callbacks import ElasticPatienceProfiler, SmartBacktrackCallback, FullTrajectoryTracker
from ltsm_core.math_utils import find_max_physical_batch, get_adaptive_batch_config

from tensorflow.keras.callbacks import ModelCheckpoint

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
            print(f"✅ В фолде [{args.fold}] уже есть модели (Используйте --append или --force).")
            return
                
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.append and MODELS_DIR.exists():
        orchestrator.sync_with_filesystem(MODELS_DIR, args.fold, Path(args.dataset_dir).name)
    
    with open(DATASET_DIR / "metadata.json", 'r', encoding='utf-8') as f:
        seq_len = json.load(f)["parameters"]["lookback"]

    with open(ARTIFACTS_DIR / "features_selected.json", 'r', encoding='utf-8') as f:
        n_features = len(json.load(f).get("feature_order", []))
            
    train_record_path = TFRECORDS_DIR / "train" / "data.tfrecord"
    val_record_path = TFRECORDS_DIR / "val" / "data.tfrecord"
    
    # -------------------------------------------------------------
    # 🧠 АДАПТИВНЫЙ БАТЧ (БЕЗОПАСНАЯ ПРОВЕРКА С БЛОКИРОВКОЙ)
    # -------------------------------------------------------------
    num_train_samples = count_tfrecord_samples(train_record_path)
    batch_config_path = ARTIFACTS_DIR / "batch_config.json"
    lock_file = ARTIFACTS_DIR / "batch_calc.lock"
    
    # Очистка "битых" замков после прерванных сессий
    if lock_file.exists():
        try: lock_file.unlink()
        except: pass

    if not batch_config_path.exists():
        try:
            # Создаем замок
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            
            print(f"\n🔍 [Worker {os.getpid()}] Начинаю тест VRAM (Остальные ждут)...")
            ideal_logical, _, _ = get_adaptive_batch_config(num_train_samples, max_phys_batch=999999)
            max_phys_batch = find_max_physical_batch(create_model, seq_len, n_features, start_batch=ideal_logical)
            logical_batch, phys_batch, accum_steps = get_adaptive_batch_config(num_train_samples, max_phys_batch)
            
            with open(batch_config_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "logical_batch": logical_batch, 
                    "phys_batch": phys_batch, 
                    "accum_steps": accum_steps
                }, f)
            
            if lock_file.exists(): 
                lock_file.unlink()
            print("✅ Конфиг батча успешно вычислен.")
                
        except FileExistsError:
            print(f"\n⏳ [Worker {os.getpid()}] Другой процесс тестирует GPU. Жду...")
            while not batch_config_path.exists():
                time.sleep(2)
            print("✅ Конфиг батча получен!")

    # Читаем финальный конфиг
    with open(batch_config_path, 'r', encoding='utf-8') as f:
        b_conf = json.load(f)
    
    logical_batch = b_conf["logical_batch"]
    phys_batch = b_conf["phys_batch"]
    accum_steps = b_conf["accum_steps"]
    
    print(f"\n♻️ Загружен размер батча: {phys_batch} (Накопление: x{accum_steps}) | Примеров: {num_train_samples}")
    
    class_weights_dict = compute_class_weights_fast(train_record_path)
    train_dataset = load_tfrecord_dataset(train_record_path, phys_batch, seq_len, n_features, is_training=True)
    val_dataset = load_tfrecord_dataset(val_record_path, phys_batch, seq_len, n_features, is_training=False)

    worker_id = f"worker_{os.getpid()}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}"
    print(f"\n🚀 Старт обучения. Воркер: {worker_id}")

    local_runs_this_session = 0

    try:
        while True:
            # 1. Читаем глобальное количество попыток из базы данных
            rows = orchestrator._execute("SELECT COUNT(*) FROM runs WHERE fold=?", (args.fold,), fetch=True)
            global_runs_done = rows[0][0] if rows else 0

            # 2. Проверка глобального бюджета Роя
            if global_runs_done >= args.runs:
                print(f"\n{'='*60}\n🛑 ГЛОБАЛЬНЫЙ БЮДЖЕТ ИСЧЕРПАН ({global_runs_done}/{args.runs} ранов).\n{'='*60}")
                sys.exit(0) # Успешный выход. Оркестратор перейдет к следующему фолду.

            current_run = global_runs_done + 1
            local_runs_this_session += 1

            can_continue, reason, global_best_loss = orchestrator.evaluate_potential(
                args.fold, worker_id, args.runs - global_runs_done, current_run_index=current_run
            )
            if not can_continue:
                print(f"\n{'='*60}\n🛑 ОСТАНОВКА ФОЛДА: {reason}\n{'='*60}")
                sys.exit(0)

            # 🌟 ДИНАМИЧЕСКИЙ ПОРОГ ДЛЯ ТОП-3
            swarm_id = os.environ.get("SWARM_ID", "manual")
            saving_threshold = orchestrator.get_saving_threshold(args.fold, keep=3)
            
            if saving_threshold == float('inf'):
                target_str = "Заполнение Топ-3 пула"
            else:
                target_str = f"Loss < {saving_threshold:.4f}"

            print(f"\n{'-'*60}\n🔄 ИТЕРАЦИЯ {current_run}/{args.runs} (Цель: {target_str})")
            print(f"📈 Статус тренда: {reason}\n{'-'*60}")
            
            run_hash = hashlib.md5(f'{time.time()}_{np.random.randint(1000)}'.encode()).hexdigest()[:6]
            run_id = f"run_{swarm_id}_{run_hash}"
            hyperparams = {"lr": args.lr, "logical_batch": logical_batch, "phys_batch": phys_batch, "l2": args.l2_reg}
            orchestrator.register_run_start(run_id, Path(args.dataset_dir).name, args.fold, hyperparams)
            
            tf.keras.backend.clear_session()
            gc.collect()
            
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            
            model = create_model(seq_len, n_features, args.l2_reg)
            
            # --- Безопасная инициализация Оптимизатора ---
            try:
                # AdamW - современный стандарт со встроенным Weight Decay
                optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=args.lr, 
                    weight_decay=args.l2_reg,
                    clipnorm=1.0  # Жестко обрубает "взрывные" векторы градиентов
                )
            except AttributeError:
                # Фолбэк для старых версий TF
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=args.lr, 
                    clipnorm=1.0
                )
                
            if accum_steps > 1:
                try:
                    optimizer = tf.keras.optimizers.experimental.GradientAccumulation(optimizer, accum_steps=accum_steps)
                    print(f"⚙️ Gradient Accumulation включен (x{accum_steps})")
                except AttributeError:
                    print(f"⚠️ Твой TF не поддерживает GradientAccumulation. Работаем на батче {phys_batch}.")
            
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                metrics=['accuracy'],
                jit_compile=True
            )
            
            temp_weights_path = MODELS_DIR / f"temp_best_{run_id}.weights.h5"
            profiler = ElasticPatienceProfiler(orchestrator, args.fold, args.epochs, args.bonus_ratio, args.min_delta)
            
            # ИСПРАВЛЕНО: Генерация пути для сохранения HDF5 файла ландшафта и инициализация трекера
            landscape_path = MODELS_DIR / f"landscape_{run_id}.h5"
            trajectory_tracker = FullTrajectoryTracker(filepath=str(landscape_path))
            
            callbacks = [
                ModelCheckpoint(filepath=temp_weights_path, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True, verbose=0),
                SmartBacktrackCallback(best_weights_path=temp_weights_path, monitor_loss='val_loss', factor=args.factor, patience=args.patience, min_lr=1e-5),
                tf.keras.callbacks.TerminateOnNaN(),
                profiler,
                trajectory_tracker # <--- ИСПРАВЛЕНО: Добавлен «бортовой самописец» в общий пул коллбеков
            ]

            try:
                history = model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=callbacks, class_weight=class_weights_dict, verbose=2)
            except KeyboardInterrupt:
                print("\n⚠️ Прервано пользователем.")
                sys.exit(1)

            if os.path.exists(temp_weights_path):
                model.load_weights(temp_weights_path)
                os.remove(temp_weights_path)
                
            loss, acc = model.evaluate(val_dataset, verbose=0)
            status = 'PRUNED' if profiler.pruned else 'COMPLETED'
            
            # 1. СНАЧАЛА узнаем проходной балл (пока база не обновилась!)
            final_threshold = orchestrator.get_saving_threshold(args.fold, keep=3)
            
            # 2. ЗАТЕМ записываем наш новый результат в базу
            orchestrator.register_run_end(
                run_id=run_id, fold_name=args.fold, val_loss=loss, val_acc=acc,
                avg_epoch_time=profiler.avg_epoch_time, overhead_time=profiler.overhead_time,
                total_ttc=profiler.total_ttc, status=status
            )
            
            # 3. Теперь проверка сработает честно (сравниваем со СТАРЫМ порогом)
            if not profiler.pruned:
                print(f"\n🎯 Итог итерации {current_run}: Val Loss = {loss:.4f} | Val Acc = {acc*100:.2f}%")
                
                if loss < final_threshold:
                    save_record_model(model, history, acc, loss, profiler.total_ttc, run_id, Path(args.dataset_dir).name, args.fold, seq_len, n_features, MODELS_DIR)
                    if loss < global_best_loss:
                        print(f"🏆 АБСОЛЮТНЫЙ РЕКОРД! Модель сохранена на 1-е место!")
                    else:
                        print(f"💎 МОДЕЛЬ ПРИНЯТА! Пробила порог Топ-3 лучших (Порог был: {final_threshold:.4f})")

            # ♻️ ПАТТЕРН "КАМИКАДЗЕ" (Считаем только локальные итерации этой жизни)
            if local_runs_this_session >= 10:
                print(f"\n♻️ [КАМИКАДЗЕ] Плановый рестарт процесса для очистки RAM (Выполнено 10 итераций подряд).")
                sys.exit(3)

    finally:
        orchestrator.remove_worker(worker_id)
        print(f"👋 Воркер {worker_id} освободил мощности.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="data/processed/2000_2026_1d")
    parser.add_argument("--fold", type=str, default="fold_2010")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=1e-5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--bonus_ratio", type=float, default=0.1)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    main(args)