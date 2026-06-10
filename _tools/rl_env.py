import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config):
        super(PortfolioTradingEnv, self).__init__()
        
        data_path = env_config["data_path"]
        df = pd.read_parquet(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # --- TRAIN / TEST SPLIT ---
        split_mode = env_config.get("split_mode", "train")
        cutoff_date = pd.to_datetime("2022-01-01") 
        
        if split_mode == "train":
            self.df_raw = df[df['datetime'] < cutoff_date].copy()
        elif split_mode == "test":
            self.df_raw = df[df['datetime'] >= cutoff_date].copy()
        else:
            self.df_raw = df.copy()
            
        # Определяем эталонный список тикеров и фичей
        self.tickers = sorted(self.df_raw['ticker'].unique().tolist())
        self.num_tickers = len(self.tickers)
        
        exclude_cols = ['datetime', 'ticker', 'open', 'high', 'low', 'close', 'close_x', 'close_y', 'volume']
        self.feature_cols = sorted([col for col in self.df_raw.columns if col not in exclude_cols])
        self.num_features = len(self.feature_cols)
        
        # --- ВЕКТОРНАЯ СБОРКА РЫНКА (PIVOTING) ---
        # Чтобы не делать циклы во время шагов, преобразуем длинную таблицу в 3D/2D тензоры
        print(f"🧬 [Portfolio Env] Векторизация рынка для {self.num_tickers} тикеров...")
        
        price_col = 'close_y' if 'close_y' in self.df_raw.columns else 'close'
        
        # Строим матрицы Цен: строки — даты, столбцы — тикеры
        self.price_pivot = self.df_raw.pivot(index='datetime', columns='ticker', values=price_col).fillna(method='ffill').fillna(method='bfill')
        self.unique_dates = self.price_pivot.index.tolist()
        self.prices_matrix = self.price_pivot.values.astype(np.float32)
        
        # Строим 3D-матрицу признаков: [Даты × Тикеры × Фичи]
        # Заполняем пропуски нулями, если какая-то бумага не торговалась в этот день
        self.features_tensor = np.zeros((len(self.unique_dates), self.num_tickers, self.num_features), dtype=np.float32)
        
        for t_idx, ticker in enumerate(self.tickers):
            ticker_df = self.df_raw[self.df_raw['ticker'] == ticker].set_index('datetime')
            # Выравниваем фичи по общей сетке дат
            ticker_df = ticker_df.reindex(self.unique_dates).fillna(0.0)
            self.features_tensor[:, t_idx, :] = ticker_df[self.feature_cols].values
            
        # --- НАСТРОЙКА RL-ПРОСТРАНСТВ ---
        # Действие: Веса портфеля для всех тикеров + 1 для Кэша
        # Box(0, 1) означает Long-Only фонд (без шортов и плеч). Сумму в 100% сделаем через Softmax.
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_tickers + 1,), dtype=np.float32
        )
        
        # Наблюдение: Все фичи всех бумаг сразу (сплющенные в 1D) + Текущие веса портфеля (включая кэш)
        self.obs_features_dim = self.num_tickers * self.num_features
        self.obs_shape = self.obs_features_dim + (self.num_tickers + 1)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )
        
        # Конфиги симуляции
        self.initial_balance = env_config.get("initial_balance", 100000.0)
        self.commission = env_config.get("commission", 0.0003)
        self.max_episode_steps = min(env_config.get("max_episode_steps", 252), len(self.unique_dates) - 2)
        
        # Переменные состояния эпизода
        self.current_step = 0
        self.episode_step = 0
        self.nav = self.initial_balance  # Чистая стоимость активов (Net Asset Value)
        self.prev_nav = self.initial_balance
        self.current_weights = np.zeros(self.num_tickers + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0  # На старте мы на 100% сидим в кэше
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        elif not hasattr(self, 'np_random'):
            self.np_random, _ = gym.utils.seeding.np_random()
            
        # Случайный выбор окна во времени для этого эпизода
        max_start = max(0, len(self.unique_dates) - self.max_episode_steps - 2)
        self.current_step = self.np_random.integers(0, max_start) if max_start > 0 else 0
        
        self.episode_step = 0
        self.nav = self.initial_balance
        self.prev_nav = self.initial_balance
        
        # Сброс в 100% кэш
        self.current_weights = np.zeros(self.num_tickers + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0
        
        return self._get_observation(), {}

    def _get_observation(self):
        # Вытаскиваем срез фичей по рынку на сегодня: матрица [Тикеры × Фичи] -> сплющиваем в 1D
        market_features = self.features_tensor[self.current_step].flatten()
        
        # Приклеиваем текущее распределение портфеля (веса)
        obs = np.concatenate([market_features, self.current_weights])
        return np.nan_to_num(obs, nan=0.0).astype(np.float32)

    def step(self, action):
        # 1. Защита и нормализация весов (Softmax), переданных агентом
        # Чтобы сумма весов была строго равна 1.0 (100% распределение)
        exp_weights = np.exp(action - np.max(action))
        target_weights = exp_weights / np.sum(exp_weights)
        
        # Цены сегодня и завтра
        prices_today = self.prices_matrix[self.current_step]
        prices_tomorrow = self.prices_matrix[self.current_step + 1]
        
        # Считаем доходность каждого тикера за шаг
        # Защита от деления на ноль, если цена почему-то пропала
        prices_today_safe = np.where(prices_today == 0, 1e-8, prices_today)
        asset_returns = (prices_tomorrow - prices_today_safe) / prices_today_safe
        
        # Добавляем доходность кэша (она всегда равна 0.0)
        portfolio_returns = np.append(asset_returns, 0.0)
        
        # 2. Расчет издержек на ребалансировку портфеля (Turnover Penalty)
        # Комиссия берется от объема сделок, необходимых для перехода от текущих весов к целевым
        weight_changes = np.sum(np.abs(target_weights - self.current_weights))
        transaction_cost = self.nav * weight_changes * self.commission
        
        # Вычитаем издержки из нашего фонда
        self.nav -= transaction_cost
        
        # 3. Переоценка активов фонда за день
        # Рост/падение фонда — это взвешенная сумма доходностей всех активов
        growth_factor = np.sum(self.current_weights * (1.0 + portfolio_returns))
        self.nav *= growth_factor
        
        # 4. Вычисляем Награду (Логарифмическая доходность фонда за шаг)
        self.nav = max(self.nav, 1.0)  # Предохранитель от полного слива
        raw_reward = np.log(self.nav / self.prev_nav) * 100.0
        step_reward = np.clip(raw_reward, -50.0, 50.0)
        
        # Обновляем состояние
        self.prev_nav = self.nav
        # Фактические веса портфеля меняются в конце дня из-за изменения цен акций
        next_weights_raw = target_weights * (1.0 + portfolio_returns)
        self.current_weights = next_weights_raw / np.sum(next_weights_raw)
        
        self.current_step += 1
        self.episode_step += 1
        
        # Критерии остановки
        terminated = False
        truncated = self.episode_step >= self.max_episode_steps
        
        # Если фонд потерял более 50% капитала — принудительный Margin Call
        if self.nav < self.initial_balance * 0.5:
            terminated = True
            step_reward -= 50.0
            
        obs = self._get_observation()
        info = {
            "balance": float(self.nav),
            "date": str(self.unique_dates[self.current_step]),
            "cash_weight": float(self.current_weights[-1])
        }
        
        return obs, float(step_reward), bool(terminated), bool(truncated), info