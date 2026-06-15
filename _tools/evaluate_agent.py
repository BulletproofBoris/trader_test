import os
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig

from _tools.rl_env import PortfolioTradingEnv
from ray.tune.registry import register_env

def env_creator(env_config):
    return PortfolioTradingEnv(env_config)

register_env("TradingEnv-v0", env_creator)

def main():
    parser = argparse.ArgumentParser(description="Инструмент оценки обученного агента.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Путь к папке чекпоинта (например, ray_results/pbt_trading_bot/PPO.../checkpoint_000000)')
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"❌ Чекпоинт {checkpoint_path} не найден!")
        return

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "processed" / "2000_2026_1d" / "rl_env" / "environment_data.parquet"
    
    env_config = {
        "data_path": str(DATA_PATH),
        "split_mode": "test", # Строго тестовая выборка
        "commission": 0.0003, # Полная комиссия
        "initial_balance": 100000.0,
        "max_episode_steps": 1000, # Чтобы пройти весь период
        "task_phase": 3 # Суровый трейдинг (Phase 3)
    }

    ray.init(ignore_reinit_error=True, logging_level="ERROR")

    print(f"🔄 Восстановление агента из {checkpoint_path}...")
    
    config = (
        PPOConfig()
        .environment("TradingEnv-v0", env_config=env_config)
        .framework("torch")
        .debugging(log_level="ERROR") 
        .training(model={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"})
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .resources(num_gpus=0, num_cpus_per_worker=1) # Оценка всегда на CPU для стабильности
    )
    
    algo = config.build()
    algo.restore(checkpoint_path)

    env = PortfolioTradingEnv(env_config)
    obs, info = env.reset()

    print(f"🚀 Запуск симуляции (Test Set)...")
    
    dates = []
    nav_history = []
    market_nav_history = []
    weights_history = []
    
    market_nav = env_config["initial_balance"]
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action = algo.compute_single_action(obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action)
        
        dates.append(info["date"])
        nav_history.append(info["balance"])
        
        market_nav *= (1.0 + info["market_return"])
        market_nav_history.append(market_nav)
        
        w = env.current_weights
        weights_history.append(w.copy())
        
    print(f"✅ Симуляция завершена. Итоговый баланс (NAV): ${nav_history[-1]:,.2f}")
    print(f"   Рыночный бенчмарк: ${market_nav_history[-1]:,.2f}")
    print(f"   Sharpe Ratio: {info.get('sharpe', 0.0):.2f}")
    print(f"   Max Drawdown: {info.get('drawdown', 0.0)*100:.2f}%")
    
    generate_report(dates, nav_history, market_nav_history, weights_history, env.tickers)

def generate_report(dates, nav_history, market_nav_history, weights_history, tickers):
    print("📈 Генерация визуального HTML-отчета...")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        subplot_titles=("Динамика Портфеля (NAV) vs Рынок", "Распределение активов (Asset Allocation)"),
                        row_heights=[0.6, 0.4])

    # 1. Графики NAV
    fig.add_trace(go.Scatter(x=dates, y=nav_history, name="RL Agent", line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=market_nav_history, name="Market Benchmark", line=dict(color='gray', dash='dash')), row=1, col=1)

    # 2. График Asset Allocation
    weights_matrix = np.array(weights_history)
    
    cash_w = weights_matrix[:, -1]
    fig.add_trace(go.Scatter(x=dates, y=cash_w, name="Cash", stackgroup='one', fillcolor='rgba(200, 200, 200, 0.5)', line=dict(width=0)), row=2, col=1)
    
    for i, ticker in enumerate(tickers):
        ticker_w = weights_matrix[:, i]
        if np.max(ticker_w) > 0.05: # Показывать в легенде только те активы, где вес превышал 5%
            fig.add_trace(go.Scatter(x=dates, y=ticker_w, name=ticker, stackgroup='one', line=dict(width=0)), row=2, col=1)

    fig.update_layout(height=800, title_text="Отчет об оценке RL Агента", hovermode="x unified")
    fig.update_yaxes(title_text="Баланс ($)", row=1, col=1)
    fig.update_yaxes(title_text="Доля портфеля", range=[0, 1], row=2, col=1)
    
    output_path = "evaluation_report.html"
    fig.write_html(output_path)
    print(f"📊 Отчет сохранен в: {Path(output_path).resolve()}")

if __name__ == "__main__":
    main()
