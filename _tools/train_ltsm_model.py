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

# Включаем BFLOAT16 (Оптимально для серии RTX 3000/4000/5000)
#tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
#print("✅ BFLOAT16 включен!")

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
from ltsm_core.callbacks import ElasticPatienceProfiler, SmartBacktrackCallback
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
    # 🧠 АДАПТИВНЫЙ БАТЧ (Hardware Limit + Math Limit)
    # -------------------------------------------------------------
    num_train_samples = count_tfrecord_samples(train_record_path)
    
    # 1. Спрашиваем математику: какой батч идеален, если бы памяти было бесконечно много? (передаем 999999)
    ideal_logical, _, _ = get_adaptive_batch_config(num_train_samples, max_phys_batch=999999)
    
    # 2. Тестируем железо, но НЕ ВЫШЕ идеального батча!
    max_phys_batch = find_max_physical_batch(create_model, seq_len, n_features, start_batch=ideal_logical)
    
    # 3. Финальный конфиг (на случай, если даже идеал не влез и нужно включить накопление градиентов)
    logical_batch, phys_batch, accum_steps = get_adaptive_batch_config(num_train_samples, max_phys_batch)
    
    print(f"\n📊 СТАТИСТИКА: {num_train_samples} тренировочных примеров.")
    print(f"🎯 Идеальный (математический) батч: {logical_batch}")
    print(f"🔧 Физический батч в VRAM: {phys_batch} (Шагов накопления: x{accum_steps})")
    
    class_weights_dict = compute_class_weights_fast(train_record_path)
    train_dataset = load_tfrecord_dataset(train_record_path, phys_batch, seq_len, n_features, is_training=True)
    val_dataset = load_tfrecord_dataset(val_record_path, phys_batch, seq_len, n_features, is_training=False)

    worker_id = f"worker_{os.getpid()}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}"
    print(f"\n🚀 Старт обучения. Воркер: {worker_id}")

    try:
        for run in range(1, args.runs + 1):
            can_continue, reason, global_best_loss = orchestrator.evaluate_potential(
                args.fold, worker_id, args.runs - run + 1, current_run_index=run
            )
            if not can_continue:
                print(f"\n{'='*60}\n🛑 ОСТАНОВКА ФОЛДА: {reason}\n{'='*60}")
                break

            # 🌟 ДИНАМИЧЕСКИЙ ПОРОГ ДЛЯ ТОП-3
            swarm_id = os.environ.get("SWARM_ID", "manual")
            saving_threshold = orchestrator.get_saving_threshold(args.fold, keep=3)
            
            # Определяем красивый вывод цели в консоль
            if saving_threshold == float('inf'):
                target_str = "Заполнение Топ-3 пула"
            else:
                target_str = f"Loss < {saving_threshold:.4f}"

            print(f"\n{'-'*60}\n🔄 ИТЕРАЦИЯ {run}/{args.runs} (Цель: {target_str})")
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
            
            # --- Накопление градиентов ---
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipvalue=0.5)
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
            
            callbacks = [
                ModelCheckpoint(filepath=temp_weights_path, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True, verbose=0),
                SmartBacktrackCallback(best_weights_path=temp_weights_path, monitor_loss='val_loss', factor=0.5, patience=3, min_lr=1e-5),
                tf.keras.callbacks.TerminateOnNaN(),
                profiler
            ]

            try:
                history = model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=callbacks, class_weight=class_weights_dict, verbose=2)
            except KeyboardInterrupt:
                print("\n⚠️ Прервано пользователем.")
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
            
            if not profiler.pruned:
                print(f"\n🎯 Итог итерации {run}: Val Loss = {loss:.4f} | Val Acc = {acc*100:.2f}%")
                
                # 🌟 ПЕРЕПРОВЕРКА ПОРОГА ПЕРЕД СОХРАНЕНИЕМ (актуально для распределенного роя)
                final_threshold = orchestrator.get_saving_threshold(args.fold, keep=3)
                
                if loss < final_threshold:
                    save_record_model(model, history, acc, loss, profiler.total_ttc, run_id, Path(args.dataset_dir).name, args.fold, seq_len, n_features, MODELS_DIR)
                    if loss < global_best_loss:
                        print(f"🏆 АБСОЛЮТНЫЙ РЕКОРД! Модель сохранена на 1-е место!")
                    else:
                        print(f"💎 МОДЕЛЯ ПРИНЯТА! Пробила порог Топ-3 лучших (Порог был: {final_threshold:.4f})")

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
    args = parser.parse_args()
    main(args)