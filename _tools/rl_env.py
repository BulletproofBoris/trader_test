import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import gc

class PortfolioTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config):
        super(PortfolioTradingEnv, self).__init__()
        
        data_path = env_config.get("data_path", "/home/restorator/trader_test/data/processed/2000_2026_1d/rl_env/environment_data.parquet")
        
        # 1. ЧИТАЕМ ДАННЫЕ ОПТИМИЗИРОВАННО
        print("🧬 [ENV] Чтение данных с диска...")
        sample_df = pd.read_parquet(data_path).head(1)
        all_cols = sample_df.columns.tolist()
        
        # Выцепляем вероятности ансамблей
        self.prob_cols = [c for c in all_cols if c.endswith('_p0') or c.endswith('_p1') or c.endswith('_p2')]
        
        # Выцепляем глобальный макро-контекст (если он есть в данных)
        potential_macro = ['usdrub_close', 'brent_close', 'sp500_close', 'imoex_close', 'vix_close']
        self.macro_cols = [c for c in potential_macro if c in all_cols]
        
        core_cols = ['datetime', 'ticker', 'close', 'close_y']
        
        # Грузим только то, что нужно агенту (без мусорных сырых фичей)
        columns_to_load = list(set(core_cols + self.prob_cols + self.macro_cols))
        df = pd.read_parquet(data_path, columns=columns_to_load)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        self.tickers = sorted(df['ticker'].unique().tolist())
        self.num_tickers = len(self.tickers)
        self.num_probs = len(self.prob_cols)
        self.num_macro = len(self.macro_cols)
        
        # --- TRAIN / TEST SPLIT ---
        split_mode = env_config.get("split_mode", "train")
        cutoff_date = pd.to_datetime("2022-01-01") 
        
        if split_mode == "train":
            df_filtered = df[df['datetime'] < cutoff_date]
        elif split_mode == "test":
            df_filtered = df[df['datetime'] >= cutoff_date]
        else:
            df_filtered = df

        print(f"🧬 [{split_mode.upper()}] Сборка State Space для {self.num_tickers} тикеров...")
        
        # А) Строим матрицу Цен
        price_col = 'close_y' if 'close_y' in df_filtered.columns else 'close'
        self.price_pivot = df_filtered.pivot(index='datetime', columns='ticker', values=price_col)
        self.price_pivot = self.price_pivot.reindex(columns=self.tickers).fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        self.unique_dates = self.price_pivot.index.tolist()
        self.prices_matrix = self.price_pivot.values.astype(np.float32)
        
        # Б) Строим тензор Вероятностей (Мнения ансамблей)
        self.probs_tensor = np.zeros((len(self.unique_dates), self.num_tickers, self.num_probs), dtype=np.float16)
        for t_idx, ticker in enumerate(self.tickers):
            if ticker in df_filtered['ticker'].values:
                ticker_df = df_filtered[df_filtered['ticker'] == ticker].set_index('datetime')
                ticker_df = ticker_df.reindex(self.unique_dates).fillna(0.0)
                ticker_df = ticker_df.reindex(columns=self.prob_cols).fillna(0.0)
                self.probs_tensor[:, t_idx, :] = ticker_df[self.prob_cols].values.astype(np.float16)
                
        # В) Строим матрицу Макро-контекста
        if self.num_macro > 0:
            # Поскольку макро одинаково для всех тикеров в один день, берем первое попавшееся значение за день
            macro_df = df_filtered.groupby('datetime')[self.macro_cols].first()
            macro_df = macro_df.reindex(self.unique_dates).fillna(0.0)
            self.macro_matrix = macro_df.values.astype(np.float32)
        else:
            self.macro_matrix = np.zeros((len(self.unique_dates), 0), dtype=np.float32)
        
        # ОЧИСТКА ПАМЯТИ
        del df, df_filtered, self.price_pivot
        gc.collect()

        # --- НАСТРОЙКА RL-ПРОСТРАНСТВ ---
        # Действия: Веса (доли) для N тикеров + 1 доля Кэша
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_tickers + 1,), dtype=np.float32
        )
        
        # Наблюдения: 
        # (Прогнозы ансамблей) + (Макро-экономика) + (Текущие доли портфеля) + (Баланс и Тайминг)
        self.obs_dim = (self.num_tickers * self.num_probs) + self.num_macro + (self.num_tickers + 1) + 2
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        self.initial_balance = env_config.get("initial_balance", 100000.0)
        self.commission = env_config.get("commission", 0.0003)
        self.max_episode_steps = min(env_config.get("max_episode_steps", 252), max(1, len(self.unique_dates) - 2))
        
        self.current_step = 0
        self.episode_step = 0
        self.nav = self.initial_balance
        self.prev_nav = self.initial_balance
        self.current_weights = np.zeros(self.num_tickers + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0  # Все деньги в кэше

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        elif not hasattr(self, 'np_random'):
            self.np_random, _ = gym.utils.seeding.np_random()
            
        max_start = max(0, len(self.unique_dates) - self.max_episode_steps - 2)
        self.current_step = self.np_random.integers(0, max_start) if max_start > 0 else 0
        
        self.episode_step = 0
        self.nav = self.initial_balance
        self.prev_nav = self.initial_balance
        
        self.current_weights = np.zeros(self.num_tickers + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0
        
        return self._get_observation(), {}

    def _get_observation(self):
        # 1. Прогнозы ансамблей
        probs_features = self.probs_tensor[self.current_step].flatten().astype(np.float32)
        
        # 2. Макро-индикаторы
        macro_features = self.macro_matrix[self.current_step]
        
        # 3. Внутренний контекст портфеля
        account_context = np.array([
            self.nav / self.initial_balance,  # Нормализованный баланс (1.0 = старт, 1.5 = +50% профит)
            self.episode_step / self.max_episode_steps # Прогресс эпизода (чтобы агент знал, когда пора "закругляться")
        ], dtype=np.float32)
        
        obs = np.concatenate([probs_features, macro_features, self.current_weights, account_context])
        return np.nan_to_num(obs, nan=0.0).astype(np.float32)

    def step(self, action):
        # Превращаем сырой вектор выхода нейросети в идеальные 100% долей (Softmax)
        exp_weights = np.exp(action - np.max(action))
        target_weights = exp_weights / np.sum(exp_weights)
        
        prices_today = self.prices_matrix[self.current_step]
        prices_tomorrow = self.prices_matrix[self.current_step + 1]
        
        prices_today_safe = np.where(prices_today == 0, 1e-8, prices_today)
        asset_returns = (prices_tomorrow - prices_today_safe) / prices_today_safe
        portfolio_returns = np.append(asset_returns, 0.0) # Кэш не меняется в цене
        
        # Комиссия за ребалансировку
        weight_changes = np.sum(np.abs(target_weights - self.current_weights))
        transaction_cost = self.nav * weight_changes * self.commission
        self.nav -= transaction_cost
        
        # Оценка стоимости активов
        growth_factor = np.sum(self.current_weights * (1.0 + portfolio_returns))
        self.nav *= growth_factor
        
        self.nav = max(self.nav, 1.0) 
        raw_reward = np.log(self.nav / self.prev_nav) * 100.0
        step_reward = np.clip(raw_reward, -50.0, 50.0)
        
        self.prev_nav = self.nav
        next_weights_raw = target_weights * (1.0 + portfolio_returns)
        self.current_weights = next_weights_raw / np.sum(next_weights_raw)
        
        self.current_step += 1
        self.episode_step += 1
        
        terminated = False
        truncated = self.episode_step >= self.max_episode_steps
        
        # Жесткий Margin Call
        if self.nav < self.initial_balance * 0.5:
            terminated = True
            step_reward -= 50.0
            
        obs = self._get_observation()
        info = {
            "balance": float(self.nav),
            "date": str(self.unique_dates[self.current_step]),
            "cash_weight": float(self.current_weights[-1])
        }
        
        if np.isnan(step_reward) or np.isinf(step_reward):
            step_reward = -10.0
            
        return obs, float(step_reward), bool(terminated), bool(truncated), info