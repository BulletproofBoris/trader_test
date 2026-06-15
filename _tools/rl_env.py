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
        
        # Выцепляем вероятности ансамблей (p0=SL, p1=Hold, p2=TP)
        self.prob_cols = [c for c in all_cols if c.endswith('_p0') or c.endswith('_p1') or c.endswith('_p2')]
        
        # Выцепляем глобальный макро-контекст (если он есть в данных)
        potential_macro = ['usdrub_close', 'brent_close', 'sp500_close', 'imoex_close', 'vix_close']
        self.macro_cols = [c for c in potential_macro if c in all_cols]
        
        # 🌟 ФИКС: Безопасный выбор базовых колонок (защита от IndexError / Missing Column)
        core_cols = ['datetime', 'ticker']
        if 'close' in all_cols: 
            core_cols.append('close')
        if 'close_y' in all_cols: 
            core_cols.append('close_y')
        
        # Грузим только то, что нужно агенту (без мусорных сырых фичей)
        columns_to_load = list(set(core_cols + self.prob_cols + self.macro_cols))
        df = pd.read_parquet(data_path, columns=columns_to_load)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 🌟 ФИКС: Глобальные размерности ДО сплита (защита от mismatch весов нейросети)
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
        # Если есть обе, 'close_y' приоритетнее (обычно это склеенный таргет), иначе 'close'
        price_col = 'close_y' if 'close_y' in df_filtered.columns else 'close'
        self.price_pivot = df_filtered.pivot(index='datetime', columns='ticker', values=price_col)
        # Принудительно расширяем матрицу до глобального списка тикеров и заполняем пустоты
        self.price_pivot = self.price_pivot.reindex(columns=self.tickers).ffill().bfill().fillna(0.0)
        self.unique_dates = self.price_pivot.index.tolist()
        self.prices_matrix = self.price_pivot.values.astype(np.float32)
        
        # Б) Строим тензор Вероятностей (Мнения ансамблей) - СЖАТО В float16 для защиты от OOM!
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
        
        # 🌟 ОЧИСТКА ПАМЯТИ (Защита от OOM)
        del df, df_filtered, self.price_pivot
        gc.collect()

        # --- НАСТРОЙКА RL-ПРОСТРАНСТВ ---
        # Действия: Веса (доли) для N тикеров + 1 доля Кэша
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_tickers + 1,), dtype=np.float32
        )
        
        # Наблюдения: 
        # (Прогнозы ансамблей) + (Макро-экономика) + (Текущие доли портфеля) + (Баланс и Тайминг)
        self.obs_dim = (self.num_tickers * self.num_probs) + self.num_macro + (self.num_tickers + 1) + 1
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        self.initial_balance = env_config.get("initial_balance", 100000.0)
        self.task_phase = env_config.get("task_phase", 1)
        
        # Настройка фаз обучения (Curriculum Learning)
        if self.task_phase == 1:
            self.commission = 0.0
            self.max_episode_steps = min(60, max(1, len(self.unique_dates) - 2))
        elif self.task_phase == 2:
            self.commission = 0.00015
            self.max_episode_steps = min(126, max(1, len(self.unique_dates) - 2))
        else:
            self.commission = env_config.get("commission", 0.0003)
            self.max_episode_steps = min(env_config.get("max_episode_steps", 252), max(1, len(self.unique_dates) - 2))
        
        self.current_step = 0

        self.episode_step = 0
        self.nav = self.initial_balance
        self.prev_nav = self.initial_balance
        self.current_weights = np.zeros(self.num_tickers + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0  # На старте сидим 100% в кэше

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
        self.peak_nav = self.initial_balance
        self.returns_history = []
        
        if self.num_macro > 0:
            self.macro_start = self.macro_matrix[self.current_step]
        
        self.current_weights = np.zeros(self.num_tickers + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0
        
        return self._get_observation(), {}

    def _get_observation(self):
        # 1. Прогнозы ансамблей
        probs_features = self.probs_tensor[self.current_step].flatten().astype(np.float32)
        
        # 2. Макро-индикаторы (Нормализованные относительно старта эпизода)
        if self.num_macro > 0:
            macro_features = (self.macro_matrix[self.current_step] - self.macro_start) / (np.abs(self.macro_start) + 1e-9)
        else:
            macro_features = np.array([], dtype=np.float32)
        
        # 3. Внутренний контекст портфеля (УБРАЛ NAV, чтобы агент не зависел от абсолютных цифр)
        account_context = np.array([
            self.episode_step / self.max_episode_steps # Только прогресс эпизода
        ], dtype=np.float32)
        
        obs = np.concatenate([probs_features, macro_features, self.current_weights, account_context])
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = np.clip(obs, -10.0, 10.0)
        return obs.astype(np.float32)

    def step(self, action):
        # 1. SOFTMAX С ИНЕРЦИЕЙ (Sticky Actions)
        # Агент выдает дельты к текущим весам, а не полностью новые веса с нуля
        raw_weights = self.current_weights[:-1] + (action[:-1] * 0.1) # Меняем максимум на 10% за шаг
        raw_cash = self.current_weights[-1] + (action[-1] * 0.1)
        
        full_raw = np.append(raw_weights, raw_cash)
        exp_weights = np.exp(full_raw - np.max(full_raw))
        target_weights = exp_weights / np.sum(exp_weights)
        
        prices_today = self.prices_matrix[self.current_step]
        prices_tomorrow = self.prices_matrix[self.current_step + 1]
        
        valid_price_mask = (prices_today > 0) & (prices_tomorrow > 0)
        asset_returns = np.zeros(self.num_tickers, dtype=np.float32)
        asset_returns[valid_price_mask] = (prices_tomorrow[valid_price_mask] - prices_today[valid_price_mask]) / prices_today[valid_price_mask]
        asset_returns = np.clip(asset_returns, -0.99, 10.0) 
        
        portfolio_returns = np.append(asset_returns, 0.0) # Кэш
        
        # 2. РАСЧЕТ БЕНЧМАРКА (Среднее по рынку за сегодня)
        # Допустим, мы вложили бы поровну во все торгующиеся бумаги
        market_return = np.mean(asset_returns[valid_price_mask]) if np.any(valid_price_mask) else 0.0
        
        # Комиссия (заниженная для стадии обучения, чтобы поощрять исследование)
        weight_changes = np.sum(np.abs(target_weights - self.current_weights))
        transaction_cost = weight_changes * (self.commission * 0.5) 
        
        # 3. ДОХОДНОСТЬ И НАГРАДА ЗА АЛЬФУ
        agent_return = np.sum(self.current_weights * portfolio_returns) - transaction_cost
        
        # НАГРАДА = Насколько агент обогнал тупой бенчмарк (в базисных пунктах)
        # Умножаем на 10000 (базисные пункты), чтобы PPO видел хорошие градиенты (например, +5.0 или -3.0)
        step_reward = (agent_return - market_return) * 10000.0 
        
        # В фазе 1 штрафуем за удержание кэша, чтобы заставить агента доверять ансамблям и торговать
        if self.task_phase == 1 and self.current_weights[-1] > 0.5:
            step_reward -= (self.current_weights[-1] * 5.0)
            
        self.returns_history.append(agent_return)
        
        # Переоценка реального NAV (для графиков)
        self.nav *= (1.0 + agent_return)
        self.nav = max(self.nav, 1.0) 
        
        self.prev_nav = self.nav
        next_weights_raw = target_weights * (1.0 + portfolio_returns)
        
        weight_sum = np.sum(next_weights_raw)
        if weight_sum > 0:
            self.current_weights = next_weights_raw / weight_sum
        else:
            self.current_weights = np.zeros_like(next_weights_raw)
            self.current_weights[-1] = 1.0 
        
        self.current_step += 1
        self.episode_step += 1
        
        terminated = False
        truncated = self.episode_step >= self.max_episode_steps
        
        if self.nav < self.initial_balance * 0.5:
            terminated = True
            step_reward -= 500.0 # Сильный штраф за Margin Call
            
        self.peak_nav = max(self.peak_nav, self.nav)
        drawdown = (self.peak_nav - self.nav) / self.peak_nav if self.peak_nav > 0 else 0.0
        
        # Штраф за Drawdown в фазе 3
        if self.task_phase >= 3 and drawdown > 0.1:
            step_reward -= (drawdown * 100.0)

        obs = self._get_observation()
        
        # Вычисление простого Sharpe
        returns_arr = np.array(self.returns_history)
        sharpe = 0.0
        if len(returns_arr) > 5 and np.std(returns_arr) > 0:
            sharpe = float(np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252))
            
        info = {
            "balance": float(self.nav),
            "date": str(self.unique_dates[self.current_step]),
            "cash_weight": float(self.current_weights[-1]),
            "agent_return": float(agent_return),
            "market_return": float(market_return),
            "drawdown": float(drawdown),
            "sharpe": sharpe
        }
        
        if np.isnan(step_reward) or np.isinf(step_reward):
            step_reward = -10.0
            
        return obs, float(step_reward), bool(terminated), bool(truncated), info