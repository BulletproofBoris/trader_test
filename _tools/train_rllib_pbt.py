import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import logging
import warnings

# --- ГЛУШИМ СИСТЕМНЫЙ СПАМ И ГИГАНТСКИЕ ТАБЛИЦЫ ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"
os.environ["RAY_TUNE_DISABLE_RICH_OUTPUT"] = "1" # Отключает огромные рамки в консоли
warnings.filterwarnings("ignore")

import ray
from ray import train, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

BASE_DIR = Path(__file__).resolve().parent.parent

# --- 1. АДАПТАЦИЯ СРЕДЫ ПОД RLLIB ---
class RLlibPortfolioEnv(gym.Env):
    def __init__(self, env_config):
        super().__init__()
        
        self.df = env_config["df"]
        self.tickers = env_config["all_tickers"]
        self.commission = env_config.get("commission", 0.0005)
        self.initial_balance = env_config.get("initial_balance", 100000.0)
        
        self.dates = sorted(self.df['datetime'].unique())
        self.num_assets = len(self.tickers)
        
        prob_cols = sorted([c for c in self.df.columns if 'prob_' in c])
        self.num_features = len(prob_cols)
        
        self.obs_tensor = np.zeros((len(self.dates), self.num_assets, self.num_features), dtype=np.float32)
        self.price_matrix = np.ones((len(self.dates), self.num_assets), dtype=np.float32)
        
        for i, date in enumerate(self.dates):
            day_data = self.df[self.df['datetime'] == date].set_index('ticker')
            for j, ticker in enumerate(self.tickers):
                if ticker in day_data.index:
                    self.obs_tensor[i, j, :] = day_data.loc[ticker, prob_cols].values
                    self.price_matrix[i, j] = day_data.loc[ticker, 'close']
                else:
                    if i > 0:
                        self.obs_tensor[i, j, :] = self.obs_tensor[i-1, j, :]
                        self.price_matrix[i, j] = self.price_matrix[i-1, j]
                        
        self.action_space = spaces.Box(low=-10, high=10, shape=(self.num_assets + 1,), dtype=np.float32)
        obs_shape = (self.num_assets * self.num_features) + (self.num_assets + 1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.weights = np.zeros(self.num_assets + 1, dtype=np.float32)
        self.weights[0] = 1.0 
        return self._get_obs(), {}

    def _get_obs(self):
        market_state = self.obs_tensor[self.current_step].flatten()
        return np.concatenate([market_state, self.weights]).astype(np.float32)

    def step(self, action):
        exp_a = np.exp(action - np.max(action)) 
        target_weights = exp_a / exp_a.sum()
        
        transaction_cost = np.sum(np.abs(target_weights - self.weights)) * self.commission
        self.weights = target_weights
        
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1
        
        if done:
            return self._get_obs(), 0.0, done, False, {}

        price_change = self.price_matrix[self.current_step] / self.price_matrix[self.current_step - 1]
        portfolio_return = self.weights[0] + np.sum(self.weights[1:] * price_change) - 1.0
        
        net_return = portfolio_return - transaction_cost
        self.balance *= (1 + net_return)
        
        reward = net_return if net_return > 0 else net_return * 2.0

        return self._get_obs(), reward, done, False, {}

# --- 2. РЕГИСТРАЦИЯ СРЕДЫ ---
def env_creator(env_config):
    return RLlibPortfolioEnv(env_config)

register_env("portfolio_env", env_creator)

# --- ИСПРАВЛЕНИЕ ОШИБОК МУТАЦИИ PBT ---
def custom_explore_fn(config):
    """Принудительно конвертируем типы и защищаем батчи от краша"""
    if "lr" in config:
        config["lr"] = float(config["lr"])
    if "entropy_coeff" in config:
        config["entropy_coeff"] = float(config["entropy_coeff"])
    if "clip_param" in config:
        config["clip_param"] = float(config["clip_param"])
        
    # Защита от ошибки кратного батча
    if "train_batch_size" in config:
        batch_size = int(config["train_batch_size"])
        config["train_batch_size"] = batch_size
        # Минибатч всегда кратен основному батчу (1/4 часть)
        config["minibatch_size"] = batch_size // 4 
        
    return config

# --- 3. НАСТРОЙКА И ЗАПУСК ЭВОЛЮЦИИ ---
def main(args):
    ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level=logging.ERROR)
    
    RL_DIR = BASE_DIR / "experiments" / "rl_trader"
    dataset_path = RL_DIR / "rl_train_dataset.csv"  
    
    print("⏳ Загрузка датасета...")
    df = pd.read_csv(dataset_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    all_tickers = sorted(df['ticker'].unique())
    
    split_date = df['datetime'].max() - pd.DateOffset(months=4)
    train_df = df[df['datetime'] < split_date]
    val_df = df[df['datetime'] >= split_date]

    # --- НАСТРОЙКА PBT ---
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        resample_probability=0.25,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-5, 1e-3),
            "entropy_coeff": tune.uniform(0.001, 0.05),
            # Только крупные батчи для плотной загрузки видеокарты
            "train_batch_size": [4096, 8192, 16384], 
            "clip_param": tune.uniform(0.1, 0.3),
        },
        custom_explore_fn=custom_explore_fn
    )

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment("portfolio_env", env_config={"df": train_df, "all_tickers": all_tickers, "commission": args.commission})
        .framework("torch")
        # 1 воркер собирает данные сразу с 8 сред (разгоняем сбор)
        .env_runners(num_env_runners=1, num_envs_per_env_runner=8) 
        # Выделяем ровно половину видеокарты (так как по CPU влезет только 2 агента)
        .resources(num_gpus=0.5) 
        .training(
            model={
                "use_lstm": True,
                "lstm_cell_size": 256,       
                "fcnet_hiddens": [512, 512], 
                "max_seq_len": 20,
            },
            lr=3e-4,
            entropy_coeff=0.01,
            train_batch_size=8192, 
            minibatch_size=2048,
            num_epochs=10,
        )
        .evaluation(
            evaluation_interval=5,
            evaluation_num_env_runners=1, 
            evaluation_config={"env_config": {"df": val_df, "all_tickers": all_tickers}}
        )
        .debugging(log_level="ERROR") 
    )

    # --- НАСТРОЙКА КОМПАКТНОГО ВЫВОДА В КОНСОЛЬ ---
    reporter = CLIReporter(
        parameter_columns={
            "lr": "Learn Rate", 
            "entropy_coeff": "Entropy",
            "train_batch_size": "Batch"
        },
        metric_columns={
            "training_iteration": "Iter",
            "env_runners/episode_return_mean": "Profit",
            "time_total_s": "Time(s)"
        },
        max_progress_rows=args.population, 
        max_report_frequency=15 
    )

    print(f"🚀 Старт PBT: Популяция из {args.population} агентов. Идет обучение...")
    
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
            name="pbt_portfolio_run",
            storage_path=str(RL_DIR / "ray_results"),
            stop={"training_iteration": args.iterations},
            verbose=1, 
            progress_reporter=reporter # Выводим нашу красивую табличку
        )
    )
    
    results = tuner.fit()
    
    if results.errors:
        print("\n❌ Во время обучения произошли ошибки! Проверьте логи в папке ray_results.")
    else:
        best_result = results.get_best_result("env_runners/episode_return_mean", "max") 
        print(f"\n🏆 Эволюция успешно завершена!")
        if best_result.checkpoint:
            print(f"Лучший агент сохранен в: {best_result.checkpoint.path}")
            print(f"Доходность лучшего агента: {best_result.metrics['env_runners']['episode_return_mean']:.2f}")
        print(f"Идеальные гиперпараметры: LR={best_result.config['lr']:.6f}, Batch={best_result.config['train_batch_size']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--population', type=int, default=4)
    parser.add_argument('--commission', type=float, default=0.0005)
    args = parser.parse_args()
    main(args)