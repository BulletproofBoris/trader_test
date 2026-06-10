import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]} # Добавлено для совместимости с gymnasium

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
        
        # 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        self.exclude_cols = ['datetime', 'ticker', 'open', 'high', 'low', 'close', 'close_x', 'close_y', 'volume']
        
        sample_df = self.grouped_data[self.tickers[0]]
        self.feature_cols = [col for col in sample_df.columns if col not in self.exclude_cols]
        
        self.prob_cols_indices = [i for i, col in enumerate(self.feature_cols) if col.endswith('_p0') or col.endswith('_p1') or col.endswith('_p2')]
        
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
        self.entry_price = 0.0 
        
    def reset(self, *, seed=None, options=None): # Добавлены * для явных kwarg аргументов
        super().reset(seed=seed)
        
        # Для корректной работы генератора случайных чисел в gymnasium
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        elif not hasattr(self, 'np_random'):
             self.np_random, _ = gym.utils.seeding.np_random()

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
        
        unrealized_pnl = 0.0
        if self.current_position != 0 and self.entry_price > 0:
            current_price = self.prices[self.current_step]
            price_change = (current_price - self.entry_price) / self.entry_price
            unrealized_pnl = price_change * self.current_position * 100.0
            
        obs = np.append(feats, [self.current_position, unrealized_pnl])
        # Строго приводим к типу, ожидаемому observation_space
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        current_price = self.prices[self.current_step]
        prev_price = self.prices[self.current_step - 1] if self.current_step > 0 else current_price
        
        desired_position = 0
        if action == 1: desired_position = 1
        elif action == 2: desired_position = -1
        
        step_reward = 0.0
        
        # --- 1. PnL (Базовая награда) ---
        if self.current_position != 0:
            daily_return = (current_price - prev_price) / prev_price
            daily_pnl_pct = daily_return * self.current_position
            self.balance *= (1 + daily_pnl_pct)
            
        # --- 2. Исполнение сделки ---
        if desired_position != self.current_position:
            commission_cost = self.commission * abs(desired_position - self.current_position)
            self.balance *= (1 - commission_cost)
            
            self.current_position = desired_position
            if desired_position != 0:
                self.entry_price = current_price
            else:
                self.entry_price = 0.0 

        # 🛡️ БЕЗОПАСНАЯ ЛОГИКА НАГРАДЫ (ПРЕДОХРАНИТЕЛЬ ОТ NaN)
        # Гарантируем, что баланс никогда не уйдет в ноль или минус
        self.balance = max(self.balance, 1.0) 
        self.prev_balance = max(self.prev_balance, 1.0)
        
        # Считаем логарифм безопасно и клипаем (обрезаем) огромные значения
        raw_log_return = np.log(self.balance / self.prev_balance) * 100.0
        step_reward = np.clip(raw_log_return, -50.0, 50.0) # Защита от "взрыва" градиентов
            
        self.prev_balance = self.balance
        self.current_step += 1
        self.episode_step += 1
        
        terminated = False
        truncated = False
        
        if self.episode_step >= self.max_episode_steps:
             truncated = True

        # Margin Call (защита от полного слива)
        if self.balance < self.initial_balance * 0.5: 
            terminated = True
            step_reward -= 20.0 
            
        # Проверяем, не выдала ли среда NaN в стейт
        obs = self._get_observation()
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        info = {
            "balance": float(self.balance), 
            "ticker": self.current_ticker,
            "position": int(self.current_position)
        }
        
        # Убеждаемся, что награда — это обычное число, а не np.float или NaN
        if np.isnan(step_reward) or np.isinf(step_reward):
            step_reward = -10.0
            
        return obs, float(step_reward), bool(terminated), bool(truncated), info