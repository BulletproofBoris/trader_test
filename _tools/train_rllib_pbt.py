import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces

import ray
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

BASE_DIR = Path(__file__).resolve().parent.parent

# --- 1. АДАПТАЦИЯ СРЕДЫ ПОД RLLIB ---
class RLlibPortfolioEnv(gym.Env):
    def __init__(self, env_config):
        super().__init__()
        
        # RLlib передает все параметры через словарь env_config
        self.df = env_config["df"]
        self.tickers = env_config["all_tickers"]
        self.commission = env_config.get("commission", 0.0005)
        self.initial_balance = env_config.get("initial_balance", 100000.0)
        
        self.dates = sorted(self.df['datetime'].unique())
        self.num_assets = len(self.tickers)
        
        prob_cols = sorted([c for c in self.df.columns if 'prob_' in c])
        self.num_features = len(prob_cols)
        
        # Чтобы не спамить в консоль от каждого воркера
        worker_idx = env_config.worker_index if hasattr(env_config, "worker_index") else 0
        if worker_idx == 1:
            print(f"Формирование 3D тензоров рынка...")
            
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

# --- 3. НАСТРОЙКА И ЗАПУСК ЭВОЛЮЦИИ ---
def main(args):
    ray.init(ignore_reinit_error=True)
    
    RL_DIR = BASE_DIR / "experiments" / "rl_trader"
    dataset_path = RL_DIR / "rl_train_dataset.csv"  
    
    print("⏳ Загрузка датасета...")
    df = pd.read_csv(dataset_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    all_tickers = sorted(df['ticker'].unique())
    
    # Разделяем датасет
    split_date = df['datetime'].max() - pd.DateOffset(months=4)
    train_df = df[df['datetime'] < split_date]
    val_df = df[df['datetime'] >= split_date]

    # --- НАСТРОЙКА PBT (Дарвиновская Эволюция) ---
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,  # Каждые 5 эпох происходит "судный день" и мутация
        resample_probability=0.25, # Шанс полной замены гиперпараметра
        hyperparam_mutations={
            # Что именно боты могут менять у себя на ходу:
            "lr": tune.loguniform(1e-5, 1e-3),
            "entropy_coeff": tune.uniform(0.001, 0.05),
            "train_batch_size": [1024, 2048, 4096],
            "clip_param": tune.uniform(0.1, 0.3),
        }
    )

    # --- БАЗОВЫЙ КОНФИГ PPO ---
    num_cpu = os.cpu_count() or 4
    workers_per_agent = max(1, (num_cpu - 1) // args.population)
    
    config = (
        PPOConfig()
        .environment("portfolio_env", env_config={"df": train_df, "all_tickers": all_tickers, "commission": args.commission})
        .framework("torch")
        # Распределение ресурсов: 
        .rollouts(num_rollout_workers=workers_per_agent) # Сбор данных
        .resources(num_gpus=1.0 / args.population) # Делим 1 видеокарту на всю популяцию
        .training(
            model={
                "use_lstm": True,
                "lstm_cell_size": 128, # Оставляем 128 для PBT, чтобы сберечь VRAM
                "max_seq_len": 20,
            },
            # Базовые стартовые параметры (будут мутировать)
            lr=3e-4,
            entropy_coeff=0.01,
            train_batch_size=2048,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
        )
        .evaluation(
            evaluation_interval=5,
            evaluation_num_workers=1,
            evaluation_config={"env_config": {"df": val_df, "all_tickers": all_tickers}}
        )
    )

    print(f"🚀 Старт PBT: Популяция из {args.population} агентов!")
    
    # Запуск планировщика Ray Tune
    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_reward_mean",
            mode="max",
            scheduler=pbt,
            num_samples=args.population, # Размер популяции
        ),
        param_space=config,
        run_config=train.RunConfig(
            name="pbt_portfolio_run",
            storage_path=str(RL_DIR / "ray_results"),
            stop={"training_iteration": args.iterations},
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=5,
                checkpoint_at_end=True
            )
        )
    )
    
    results = tuner.fit()
    best_result = results.get_best_result("env_runners/episode_reward_mean", "max")
    
    print(f"\n🏆 Эволюция завершена!")
    print(f"Лучший агент сохранен в: {best_result.checkpoint.path}")
    print(f"Его идеальные гиперпараметры: {best_result.config['lr']}, Entropy: {best_result.config['entropy_coeff']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--population', type=int, default=4, help="Количество конкурирующих ботов")
    parser.add_argument('--commission', type=float, default=0.0005)
    args = parser.parse_args()
    main(args)