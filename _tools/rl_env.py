import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, env_config):
        super(TradingEnv, self).__init__()
        
        # --- 1. ЗАЩИТА ПАМЯТИ: Читаем данные по пути ---
        data_path = env_config["data_path"]
        df = pd.read_parquet(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # --- 2. TRAIN / TEST SPLIT ---
        split_mode = env_config.get("split_mode", "train")
        cutoff_date = pd.to_datetime("2022-01-01") # Рубикон
        
        if split_mode == "train":
            # Обучаемся на спокойном и умеренно-волатильном рынке (до 2022)
            self.df = df[df['datetime'] < cutoff_date].copy()
        elif split_mode == "test":
            # Проверяем чемпиона в условиях СВО, санкций и высоких ставок (2022+)
            self.df = df[df['datetime'] >= cutoff_date].copy()
        else:
            self.df = df.copy() # На случай полного прогона
            
        self.initial_balance = env_config.get("initial_balance", 100000.0)
        self.commission = env_config.get("commission", 0.0003)
        self.max_episode_steps = env_config.get("max_episode_steps", 252) # 1 год торговли
        
        self.action_space = spaces.Discrete(3)
        self.exclude_cols = ['datetime', 'ticker', 'open', 'high', 'low', 'close_x', 'close_y', 'volume']
        self.feature_cols = [col for col in self.df.columns if col not in self.exclude_cols]
        
        self.obs_shape = len(self.feature_cols) + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )
        
        self.prices = self.df['close_y'].values.astype(np.float32)
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.total_steps = len(self.prices) - 1
        
        # Переменные состояния
        self.current_step = 0
        self.episode_step = 0
        self.balance = self.initial_balance
        self.current_position = 0 
        self.entry_price = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Выбираем случайную стартовую точку в датасете
        max_start = max(0, self.total_steps - self.max_episode_steps - 1)
        self.current_step = self.np_random.integers(0, max_start) if max_start > 0 else 0
        
        self.episode_step = 0
        self.balance = self.initial_balance
        self.current_position = 0
        self.entry_price = 0.0
        
        return self._get_observation(), {}

    def _get_observation(self):
        feats = self.features[self.current_step]
        current_price = self.prices[self.current_step]
        
        unrealized_pnl = 0.0
        if self.current_position != 0 and self.entry_price > 0:
            unrealized_pnl = ((current_price - self.entry_price) / self.entry_price) * self.current_position
            
        obs = np.append(feats, [self.current_position, unrealized_pnl])
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.prices[self.current_step]
        desired_position = action - 1 
        reward = 0.0
        step_penalty = 0.0001
        
        if desired_position != self.current_position:
            if self.current_position != 0:
                pnl = ((current_price - self.entry_price) / self.entry_price) * self.current_position
                pnl -= self.commission
                reward = pnl * 100 
                self.balance *= (1 + pnl)
                
            if desired_position != 0:
                self.entry_price = current_price
                reward -= self.commission * 100
                
            self.current_position = desired_position
        else:
            reward -= step_penalty
            if self.current_position != 0:
                unrealized = ((current_price - self.entry_price) / self.entry_price) * self.current_position
                reward += unrealized * 0.01 * 100

        self.current_step += 1
        self.episode_step += 1
        
        terminated = False
        truncated = bool(self.episode_step >= self.max_episode_steps)
        
        if self.balance < self.initial_balance * 0.1: 
            terminated = True
            reward -= 10.0 # Штраф за маржин-колл
            
        obs = self._get_observation()
        info = {"balance": self.balance}
        
        return obs, reward, terminated, truncated, info