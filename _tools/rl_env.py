import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, env_config):
        super(TradingEnv, self).__init__()
        
        data_path = env_config["data_path"]
        df = pd.read_parquet(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # --- TRAIN / TEST SPLIT ---
        split_mode = env_config.get("split_mode", "train")
        cutoff_date = pd.to_datetime("2022-01-01") 
        
        if split_mode == "train":
            base_df = df[df['datetime'] < cutoff_date].copy()
        elif split_mode == "test":
            base_df = df[df['datetime'] >= cutoff_date].copy()
        elif split_mode == "2025_2026":
            base_df = df[df['datetime'] >= pd.to_datetime("2025-01-01")].copy()
        elif split_mode == "2026":
            base_df = df[df['datetime'] >= pd.to_datetime("2026-01-01")].copy()
        else:
            base_df = df.copy() 
            
        # --- ГРУППИРОВКА ПО ТИКЕРАМ ---
        self.grouped_data = {}
        for ticker, group in base_df.groupby('ticker'):
            if len(group) > env_config.get("max_episode_steps", 252):
                self.grouped_data[ticker] = group.sort_values('datetime').reset_index(drop=True)
                
        self.tickers = list(self.grouped_data.keys())
        
        if not self.tickers:
            raise ValueError(f"После фильтрации '{split_mode}' не осталось тикеров с достаточной историей!")
            
        self.initial_balance = env_config.get("initial_balance", 100000.0)
        self.commission = env_config.get("commission", 0.0003)
        self.max_episode_steps = env_config.get("max_episode_steps", 252) 
        
        self.action_space = spaces.Discrete(3)
        self.exclude_cols = ['datetime', 'ticker', 'open', 'high', 'low', 'close', 'close_x', 'close_y', 'volume']
        
        sample_df = self.grouped_data[self.tickers[0]]
        self.feature_cols = [col for col in sample_df.columns if col not in self.exclude_cols]
        
        # Размерность: Фичи + Текущая позиция (-1,0,1) + Unrealized PnL %
        self.obs_shape = len(self.feature_cols) + 2 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )
        
        self.df = None
        self.prices = None
        self.features = None
        self.total_steps = 0
        
        self.current_step = 0
        self.episode_step = 0
        self.balance = self.initial_balance
        self.prev_balance = self.initial_balance 
        self.current_position = 0 
        self.current_ticker = None
        self.entry_price = 0.0 
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_ticker = self.np_random.choice(self.tickers)
        self.df = self.grouped_data[self.current_ticker]
        
        price_col = 'close_y' if 'close_y' in self.df.columns else 'close'
        self.prices = self.df[price_col].values.astype(np.float32)
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.total_steps = len(self.prices) - 1
        
        max_start = max(0, self.total_steps - self.max_episode_steps - 1)
        self.current_step = self.np_random.integers(0, max_start) if max_start > 0 else 0
        
        self.episode_step = 0
        self.balance = self.initial_balance
        self.prev_balance = self.initial_balance
        self.current_position = 0
        self.entry_price = 0.0
        
        return self._get_observation(), {}

    def _get_observation(self):
        feats = self.features[self.current_step]
        
        # Считаем нереализованную прибыль/убыток в процентах
        unrealized_pnl = 0.0
        if self.current_position != 0 and self.entry_price > 0:
            current_price = self.prices[self.current_step]
            price_change = (current_price - self.entry_price) / self.entry_price
            unrealized_pnl = price_change * self.current_position * 100.0 # В процентах
            
        obs = np.append(feats, [self.current_position, unrealized_pnl])
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.prices[self.current_step]
        prev_price = self.prices[self.current_step - 1] if self.current_step > 0 else current_price
        
        desired_position = action - 1 
        
        # 1. Считаем PnL от удержания позиции
        if self.current_position != 0:
            daily_return = (current_price - prev_price) / prev_price
            daily_pnl_pct = daily_return * self.current_position
            self.balance *= (1 + daily_pnl_pct)
            
        # 2. Обработка изменения позиции (Комиссии и Цена входа)
        if desired_position != self.current_position:
            commission_cost = self.commission * abs(desired_position - self.current_position)
            self.balance *= (1 - commission_cost)
            self.current_position = desired_position
            
            if desired_position != 0:
                self.entry_price = current_price
            else:
                self.entry_price = 0.0 

        # 3. Награда агента (Исправлено на Логарифмическую доходность)
        # Логарифм делает награду симметричной для PPO-сетей
        if self.balance > 0 and self.prev_balance > 0:
            step_reward = np.log(self.balance / self.prev_balance) * 100.0
        else:
            step_reward = -100.0 # Катастрофа, слив
        
        # Штраф за нахождение в кэше удален. Агент должен уметь ждать!
            
        self.prev_balance = self.balance
        self.current_step += 1
        self.episode_step += 1
        
        terminated = False
        truncated = bool(self.episode_step >= self.max_episode_steps)
        
        # Защита от слива депозита (Margin Call)
        if self.balance < self.initial_balance * 0.1: 
            terminated = True
            step_reward -= 10.0 
            
        obs = self._get_observation()
        info = {
            "balance": self.balance, 
            "ticker": self.current_ticker,
            "position": self.current_position
        }
        
        return obs, step_reward, terminated, truncated, info