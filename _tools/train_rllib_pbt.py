import os
import argparse
import logging
import warnings
import sys
import csv
from datetime import datetime
import shutil
from pathlib import Path
import numpy as np

# =====================================================================
# НАСТРОЙКИ СИСТЕМЫ И ПАМЯТИ (ЗАЩИТА ОТ OOM)
# =====================================================================
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "0"

# 🌟 ФИКС 1: Запрещаем PyTorch жадно резервировать видеопамять
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

# 1. Парсим аргументы ДО инициализации тяжелых библиотек
parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=500) 
parser.add_argument('--population', type=int, default=4)
parser.add_argument('--force', action='store_true', help='Принудительно начать обучение с нуля')
parser.add_argument('--cpu', action='store_true', help='Отключить GPU и учить только на CPU')
args = parser.parse_args()
TOTAL_ITERATIONS = args.iterations

# 2. Управление ресурсами
if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    NUM_GPUS = 0
    print("🖥️ РЕЖИМ CPU: Видеокарты принудительно отключены.")
else:
    NUM_GPUS = 1  # Если у тебя RTX 5070
    print("🎮 РЕЖИМ GPU: Использование видеокарты разрешено.")

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune import CLIReporter
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.train import CheckpointConfig

from _tools.rl_env import PortfolioTradingEnv

BASE_DIR = Path(__file__).resolve().parent.parent
RL_DIR = BASE_DIR / "data" / "processed" / "2000_2026_1d" / "rl_env"
STATS_FILE = RL_DIR / "training_summary.txt"
CSV_LOG_FILE = RL_DIR / "training_progress.csv"

# --- КАЛБЭК: Запись статистики в CSV и TXT ---
class TradingStatsCallback(tune.Callback):
    def __init__(self):
        super().__init__()
        # Инициализируем CSV файл и пишем заголовки, если его еще нет
        if not CSV_LOG_FILE.exists() or args.force:
            with open(CSV_LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Iteration", "Phase", "Trial_ID", "Status", "Train_Return", "Test_Return", "Train_Sharpe"])

    def on_trial_result(self, iteration, trials, trial, result, **info):
        lines = []
        lines.append("="*60)
        lines.append(f"📊 ОБНОВЛЕНИЕ СТАТИСТИКИ (Итерация {result.get('training_iteration', 0)})")
        lines.append("="*60)
        lines.append(f"{'Trial ID':<15} | {'Phase':<5} | {'Train Ret %':<12} | {'Test Ret %':<12} | {'Sharpe':<8}")
        lines.append("-" * 60)

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_data_to_append = []

        for t in trials:
            m = t.last_result
            if not m:
                continue
                
            train_ret = m.get('env_runners', {}).get('episode_return_mean', 0)
            eval_ret = m.get('evaluation', {}).get('env_runners', {}).get('episode_return_mean', np.nan)
            
            # Извлекаем кастомные метрики (Sharpe)
            custom_metrics = m.get('custom_metrics', {})
            train_sharpe = custom_metrics.get('sharpe_mean', 0.0)
            current_phase = custom_metrics.get('task_phase_mean', 1.0)
            
            # Строки для TXT
            eval_str = f"{eval_ret:.2f}" if not np.isnan(eval_ret) else "WAITING"
            lines.append(f"{t.trial_id:<15} | {int(current_phase):<5} | {train_ret:<12.2f} | {eval_str:<12} | {train_sharpe:<8.2f}")
            
            # Строки для CSV
            csv_data_to_append.append([
                current_time, 
                m.get('training_iteration', 0), 
                int(current_phase),
                t.trial_id, 
                t.status, 
                round(train_ret, 4), 
                round(eval_ret, 4) if not np.isnan(eval_ret) else "",
                round(train_sharpe, 4)
            ])

        lines.append("\n* Train Ret: Альфа-доходность портфеля (обгон бенчмарка в б.п.)")
        lines.append("* Test Ret:  Средняя награда на экзаменационных данных (2022-2024)")
        
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            
        if csv_data_to_append:
            with open(CSV_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data_to_append)

# --- КАЛБЭК: Отслеживание метрик и Curriculum Learning ---
class CustomMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        info = episode.last_info_for()
        if info:
            if "sharpe" in info:
                episode.custom_metrics["sharpe"] = info["sharpe"]
            if "drawdown" in info:
                episode.custom_metrics["drawdown"] = info["drawdown"]
            episode.custom_metrics["task_phase"] = getattr(worker.env, "task_phase", 1)

    def on_train_result(self, *, algorithm, result, **kwargs):
        iteration = result["training_iteration"]
        
        # Определяем фазу (Curriculum Learning)
        phase = 1
        if iteration > TOTAL_ITERATIONS / 5:
            phase = 2
        if iteration > TOTAL_ITERATIONS / 2:
            phase = 3
            
        # Безопасная рассылка новой фазы
        env_group = getattr(algorithm, "env_runner_group", getattr(algorithm, "workers", None))
        if callable(env_group):
            env_group = env_group() 
            
        if env_group is not None:
            env_group.foreach_env(lambda env: setattr(env, 'task_phase', phase))
            
        # Также обновляем фазу в evaluation воркерах
        eval_group = getattr(algorithm, "eval_env_runner_group", getattr(algorithm, "evaluation_workers", None))
        if callable(eval_group):
            eval_group = eval_group()
            
        if eval_group is not None:
            eval_group.foreach_env(lambda env: setattr(env, 'task_phase', phase))

def env_creator(env_config):
    return PortfolioTradingEnv(env_config)

register_env("TradingEnv-v0", env_creator)

def main():
    DATA_PATH = RL_DIR / "environment_data.parquet"
    EXPERIMENT_NAME = "pbt_trading_bot"
    EXPERIMENT_DIR = RL_DIR / "ray_results" / EXPERIMENT_NAME
    
    if args.force:
        if EXPERIMENT_DIR.exists():
            print(f"⚠️ Флаг --force обнаружен. Удаляем старые результаты...")
            shutil.rmtree(EXPERIMENT_DIR, ignore_errors=True)
            if STATS_FILE.exists():
                os.remove(STATS_FILE)
            print("✅ Старые данные удалены. Начинаем с чистого листа.")

    env_config = {
        "data_path": str(DATA_PATH),
        "split_mode": "train",
        "commission": 0.0003,
        "initial_balance": 100000.0,
        "max_episode_steps": 252
    }
    
    # 🌟 ФИКС 2: Жестко ограничиваем разделяемую память (Object Store)
    ray.init(
        ignore_reinit_error=True, 
        logging_level=logging.ERROR, 
        num_gpus=NUM_GPUS,
        object_store_memory=2 * 1024 * 1024 * 1024  # 2 GB
    )

    config = (
        PPOConfig()
        .environment("TradingEnv-v0", env_config=env_config)
        .framework("torch")
        .debugging(log_level="ERROR") 
        .training(
            lr=1e-4,
            train_batch_size=4096, 
            model={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"}
        )
        .callbacks(CustomMetricsCallback)
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .resources(
            num_gpus=NUM_GPUS / args.population if NUM_GPUS > 0 else 0, 
            num_cpus_per_worker=1,
            num_gpus_per_worker=0
        )
        # 🌟 ФИКС 3: Запрещаем дублировать среду на главном узле (уменьшает потребление RAM)
        .env_runners(
            num_env_runners=1,
            create_env_on_local_worker=False
        )
        .evaluation(
            evaluation_interval=10, 
            evaluation_duration=5, 
            evaluation_config={"env_config": {"split_mode": "test"}, "explore": False}
        )
    )

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=30, 
        resample_probability=0.25,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-5, 5e-4),
            "entropy_coeff": tune.uniform(0.0, 0.05),
            "vf_loss_coeff": tune.uniform(0.1, 1.0)
        }
    )

    reporter = CLIReporter(
        metric_columns=["training_iteration", "env_runners/episode_return_mean"],
        max_progress_rows=1,
        print_intermediate_tables=False
    )
    
    # 🌟 ФИКС 4 (RAY BUG): Создаем конфиг без указания устаревших параметров
    ckpt_config = CheckpointConfig(
        num_to_keep=3,
        checkpoint_score_attribute="env_runners/episode_return_mean",
        checkpoint_score_order="max"
    )
    
    # 🌟 ВЗЛОМ ОБЪЕКТА: Принудительно вшиваем параметры в обход проверок версии Ray Train V2
    ckpt_config.checkpoint_frequency = 10
    ckpt_config.checkpoint_at_end = True
    
    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_return_mean",
            mode="max",
            scheduler=pbt,
            num_samples=args.population,
        ),
        param_space=config,
        run_config=tune.RunConfig(
            name=EXPERIMENT_NAME,
            storage_path=str(RL_DIR / "ray_results"),
            checkpoint_config=ckpt_config,
            callbacks=[TradingStatsCallback()], 
            progress_reporter=reporter, 
            verbose=1, 
            stop={"training_iteration": args.iterations} 
        )
    )
    
    can_fit = True
    if not args.force and tune.Tuner.can_restore(str(EXPERIMENT_DIR)):
        try:
            print("🔄 Восстановление сессии обучения...")
            tuner = tune.Tuner.restore(str(EXPERIMENT_DIR), trainable="PPO", resume_errored=True)
        except Exception as e:
            print(f"⚠️ Не удалось восстановить сессию: {e}")
            can_fit = False

    if can_fit:
        try:
            print("⏳ Обучение запущено. Открой файл training_progress.csv для просмотра метрик.")
            tuner.fit()
        except KeyboardInterrupt:
            print("\n🛑 Остановка по Ctrl+C...")
    else:
         print("ℹ️ Обучение уже завершено или остановлено.")

if __name__ == "__main__":
    main()