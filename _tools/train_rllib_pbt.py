import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import logging
import warnings

# --- ГЛУШИМ СИСТЕМНЫЙ СПАМ ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"
os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"
warnings.filterwarnings("ignore")

import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Импортируем нашу новую среду
from _tools.rl_env import TradingEnv

BASE_DIR = Path(__file__).resolve().parent.parent

# --- 1. АДАПТАЦИЯ СРЕДЫ ПОД RLLIB ---
def env_creator(env_config):
    # Теперь просто прокидываем словарь настроек в среду
    return TradingEnv(env_config)

register_env("TradingEnv-v0", env_creator)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=50, help='Количество поколений эволюции')
    parser.add_argument('--population', type=int, default=4, help='Размер популяции агентов')
    args = parser.parse_args()

    RL_DIR = BASE_DIR / "data" / "processed" / "2000_2026_1d" / "rl_env"
    DATA_PATH = RL_DIR / "environment_data.parquet"
    
    if not DATA_PATH.exists():
        print(f"❌ Файл с данными не найден: {DATA_PATH}")
        return

    # МЫ БОЛЬШЕ НЕ ГРУЗИМ DATAFRAME ЗДЕСЬ! Защита от OOM (утечки памяти)
    env_config = {
        "data_path": str(DATA_PATH),
        "split_mode": "train", # Основные воркеры учатся на периоде до 2022 года
        "commission": 0.0003,
        "initial_balance": 100000.0,
        "max_episode_steps": 252
    }

    print("🚀 Инициализация Ray Cluster...")
    ray.init(ignore_reinit_error=True, logging_level=logging.ERROR)

    # --- 2. БАЗОВАЯ КОНФИГУРАЦИЯ PPO ---
    config = (
        PPOConfig()
        .environment("TradingEnv-v0", env_config=env_config)
        .framework("torch")
        .training(
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            lr=1e-4,
            train_batch_size=1024,
            minibatch_size=128,
            entropy_coeff=0.01,
            clip_param=0.2,
        )
        .env_runners(
            num_env_runners=1,
            rollout_fragment_length=1024
        )
        # ======= 3. МАГИЯ ТЕСТИРОВАНИЯ (Out-of-Sample) =======
        .evaluation(
            evaluation_interval=5,    # Каждые 5 итераций делаем экзамен
            evaluation_duration=5,    # Длительность экзамена: 5 лет/эпизодов
            evaluation_config={
                "env_config": {
                    "split_mode": "test" # Среда экзамена грузит данные после 2022 года
                },
                "explore": False      # На экзамене агент НЕ экспериментирует, только торгует
            }
        )
        .resources(num_gpus=1 if ray.cluster_resources().get("GPU", 0) > 0 else 0)
    )

    # --- 4. НАСТРОЙКА ЭВОЛЮЦИИ (PBT) ---
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        resample_probability=0.25,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-5, 1e-3),
            "entropy_coeff": tune.uniform(0.001, 0.05),
            "clip_param": tune.uniform(0.1, 0.3),
        },
        custom_explore_fn=None,
    )

    print(f"\n🧬 Запуск Population-Based Training (PBT)...")
    print(f"   Популяция: {args.population} агентов")
    print(f"   Поколений: {args.iterations} (можно остановить вручную)\n")

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
            name="pbt_trading_bot",
            storage_path=str(RL_DIR / "ray_results"),
        )
    )
    
    print("\n⚠️ ВНИМАНИЕ: Алгоритм будет эволюционировать.")
    print("Когда увидите в таблице хорошую прибыль, нажмите STOP в Jupyter (или Ctrl+C в терминале).")
    print("Скрипт перехватит сигнал и безопасно достанет лучшего агента с диска!\n")

    try:
        results = tuner.fit()
    except KeyboardInterrupt:
        print("\n\n🛑 Остановка эволюции пользователем! Извлекаем лучших мутантов с диска...")
        restored_tuner = tune.Tuner.restore(str(RL_DIR / "ray_results" / "pbt_trading_bot"))
        results = restored_tuner.get_results()
    
    if results.errors:
        print("\n❌ Во время обучения произошли ошибки! Проверьте логи в папке ray_results.")
    else:
        best_result = results.get_best_result("env_runners/episode_return_mean", "max") 
        print(f"\n🏆 Эволюция успешно завершена!")
        if best_result and best_result.checkpoint:
            print(f"Лучший агент сохранен в: {best_result.checkpoint.path}")
            print(f"Доходность лучшего агента (Train Reward Mean): {best_result.metrics.get('env_runners', {}).get('episode_return_mean', 0):.4f}")
        print(f"Идеальные гиперпараметры: LR={best_result.config['lr']:.6f}, Entropy={best_result.config['entropy_coeff']:.4f}")

if __name__ == "__main__":
    main()