import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
import sys
import shutil # Нужен для удаления папок

# Глушим системный спам
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
warnings.filterwarnings("ignore")

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune import CLIReporter

# Импортируем нашу среду
from _tools.rl_env import TradingEnv

BASE_DIR = Path(__file__).resolve().parent.parent
RL_DIR = BASE_DIR / "data" / "processed" / "2000_2026_1d" / "rl_env"
STATS_FILE = RL_DIR / "training_summary.txt"

# --- CALLBACK ДЛЯ ЗАПИСИ СВОИХ ЛОГОВ В ФАЙЛ ---
class TradingStatsCallback(tune.Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        lines = []
        lines.append("="*60)
        lines.append(f"📊 ОБНОВЛЕНИЕ СТАТИСТИКИ (Итерация {result.get('training_iteration', 0)})")
        lines.append("="*60)
        lines.append(f"{'Trial ID':<15} | {'Status':<10} | {'Train Ret %':<12} | {'Test Ret %':<12}")
        lines.append("-"*60)

        for t in trials:
            m = t.last_result
            if not m: # Защита от пустых результатов на старте
                continue
                
            train_ret = m.get('env_runners', {}).get('episode_return_mean', 0)
            eval_ret = m.get('evaluation', {}).get('env_runners', {}).get('episode_return_mean', np.nan)
            eval_str = f"{eval_ret:.2f}" if not np.isnan(eval_ret) else "WAITING..."
            
            lines.append(f"{t.trial_id:<15} | {t.status:<10} | {train_ret:<12.2f} | {eval_str:<12}")

        lines.append("\n* Train Ret: доходность на данных до 2022 года")
        lines.append("* Test Ret:  доходность на данных 2022-2024 (экзамен)")
        
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

def env_creator(env_config):
    return TradingEnv(env_config)

register_env("TradingEnv-v0", env_creator)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--population', type=int, default=4)
    # --- ДОБАВЛЯЕМ НОВЫЙ АРГУМЕНТ ---
    parser.add_argument('--force', action='store_true', help='Принудительно начать обучение с нуля, удалив старые модели')
    args = parser.parse_args()

    DATA_PATH = RL_DIR / "environment_data.parquet"
    EXPERIMENT_NAME = "pbt_trading_bot"
    EXPERIMENT_DIR = RL_DIR / "ray_results" / EXPERIMENT_NAME
    
    # --- ЛОГИКА ПРИНУДИТЕЛЬНОГО ПЕРЕОБУЧЕНИЯ ---
    if args.force:
        if EXPERIMENT_DIR.exists():
            print(f"⚠️ Флаг --force обнаружен. Удаляем старые результаты из {EXPERIMENT_DIR}...")
            shutil.rmtree(EXPERIMENT_DIR, ignore_errors=True)
            if STATS_FILE.exists():
                os.remove(STATS_FILE)
            print("✅ Старые данные удалены. Начинаем с чистого листа.")
        else:
            print("ℹ️ Флаг --force передан, но старых данных нет. Начинаем обучение.")

    env_config = {
        "data_path": str(DATA_PATH),
        "split_mode": "train",
        "commission": 0.0003,
        "initial_balance": 100000.0,
        "max_episode_steps": 252
    }

    print(f"🚀 Инициализация. Логи будут здесь: {STATS_FILE}")
    ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)

    config = (
        PPOConfig()
        .environment("TradingEnv-v0", env_config=env_config)
        .framework("torch")
        .debugging(log_level="ERROR") 
        .training(
            lr=1e-4,
            train_batch_size=1024,
            model={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"}
        )
        .env_runners(num_env_runners=1)
        .evaluation(
            evaluation_interval=2, 
            evaluation_duration=3, 
            evaluation_config={"env_config": {"split_mode": "test"}, "explore": False}
        )
    )

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        resample_probability=0.25,
        hyperparam_mutations={"lr": tune.loguniform(1e-5, 1e-3)}
    )

    reporter = CLIReporter(
        metric_columns=["training_iteration", "env_runners/episode_return_mean"],
        max_progress_rows=1,
        print_intermediate_tables=False
    )

    # --- ЛОГИКА ВОССТАНОВЛЕНИЯ ИЛИ НОВОГО СТАРТА ---
    # Если папка существует и мы НЕ передали --force, Ray попытается продолжить
    # Если папки нет (мы ее удалили выше), он создаст новый
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
            callbacks=[TradingStatsCallback()], 
            progress_reporter=reporter, 
            verbose=1, 
            stop={"training_iteration": args.iterations} 
        )
    )
    
    # Если мы не используем --force, нужно проверить, не закончено ли уже обучение
    # Иначе Tuner попытается продолжить и вылетит с ошибкой "Tuner is already done"
    can_fit = True
    if not args.force and tune.Tuner.can_restore(str(EXPERIMENT_DIR)):
        try:
            print("🔄 Попытка восстановить предыдущую сессию обучения...")
            tuner = tune.Tuner.restore(str(EXPERIMENT_DIR), trainable="PPO", resume_errored=True)
            # Устанавливаем новый лимит итераций, если пользователь передал большее число
            # Это обходной путь для Ray 2.x, так как run_config.stop жестко запекается
        except Exception as e:
            print(f"⚠️ Не удалось восстановить сессию: {e}. Возможно, она уже завершена.")
            can_fit = False

    if can_fit:
        try:
            print("⏳ Обучение запущено...")
            tuner.fit()
        except KeyboardInterrupt:
            print("\n🛑 Остановка по Ctrl+C...")
    else:
         print(f"ℹ️ Обучение уже завершено или остановлено. Если хочешь начать заново, используй флаг --force")

if __name__ == "__main__":
    main()