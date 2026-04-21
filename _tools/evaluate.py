import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

# Настройки среды
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"
warnings.filterwarnings("ignore")

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

import sys
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from train_rllib_pbt import RLlibPortfolioEnv, env_creator

def run_evaluation():
    # Настройки путей
    CHECKPOINT = "/home/restorator/trader_test/experiments/rl_trader/ray_results/pbt_portfolio_run/PPO_portfolio_env_83d81_00001_1_2026-04-20_12-51-33/checkpoint_000017"
    DATASET_PATH = Path("/home/restorator/trader_test/experiments/rl_trader/rl_test_dataset.csv")
    OUTPUT_DIR = Path("/home/restorator/trader_test/experiments/rl_trader")
    
    INITIAL_BALANCE = 100000.0
    COMMISSION = 0.0005

    print(f"📂 Загрузка тестовых данных (2026 год)...")
    df = pd.read_csv(DATASET_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # ФИКС: Берем эталонный список тикеров из файла обучения, чтобы нейросеть не сломалась!
    TRAIN_PATH = Path("/home/restorator/trader_test/experiments/rl_trader/rl_train_dataset.csv")
    train_df = pd.read_csv(TRAIN_PATH)
    all_tickers = sorted(train_df['ticker'].unique())
    
    test_df = df.copy() 
    test_dates = sorted(test_df['datetime'].unique())

    # Инициализация
    ray.init(ignore_reinit_error=True, logging_level="ERROR")
    register_env("portfolio_env", env_creator)
    
    print(f"🧠 Загрузка чекпоинта: {Path(CHECKPOINT).name}")
    algo = Algorithm.from_checkpoint(CHECKPOINT)

    env_config = {
        "df": test_df, 
        "all_tickers": all_tickers, 
        "commission": COMMISSION,
        "initial_balance": INITIAL_BALANCE
    }
    env = RLlibPortfolioEnv(env_config)

    # Симуляция
    print(f"📊 Бэктест на {len(test_dates)} днях ({test_dates[0].date()} - {test_dates[-1].date()})...")
    obs, _ = env.reset()
    done = False
    
    # Списки для визуализации и детального лога
    results = []
    detailed_log = []
    
    lstm_state = algo.get_policy().get_initial_state()
    
    while not done:
        prev_balance = env.balance
        
        action, lstm_state, _ = algo.compute_single_action(
            observation=obs, 
            state=lstm_state, 
            explore=False
        )
        obs, reward, done, _, _ = env.step(action)
        
        current_date = test_dates[env.current_step - 1].date()
        daily_return_pct = ((env.balance / prev_balance) - 1) * 100 if prev_balance > 0 else 0.0
        
        # 1. Данные для графиков
        results.append({
            "date": current_date,
            "balance": env.balance,
            "weights": env.weights.copy()
        })
        
        # 2. Данные для подробной таблицы
        log_row = {
            "Date": current_date,
            "Balance (RUB)": round(env.balance, 2),
            "Daily Return (%)": round(daily_return_pct, 4),
            "RL Reward": round(reward, 6),
            "CASH (%)": round(env.weights[0] * 100, 2)
        }
        # Добавляем доли каждой акции в этот день
        for i, ticker in enumerate(all_tickers):
            log_row[f"{ticker} (%)"] = round(env.weights[i+1] * 100, 2)
            
        detailed_log.append(log_row)

    ray.shutdown()
    
    # --- СОХРАНЕНИЕ ТАБЛИЦЫ ---
    log_df = pd.DataFrame(detailed_log)
    log_file_path = OUTPUT_DIR / "backtest_detailed_log.csv"
    log_df.to_csv(log_file_path, index=False)
    print(f"📁 Подробный лог операций сохранен в: {log_file_path}")

    # --- ФОРМИРОВАНИЕ ТЕКСТОВОГО ОТЧЕТА ---
    res_df = pd.DataFrame(results)
    final_balance = res_df['balance'].iloc[-1]
    profit_pct = ((final_balance / INITIAL_BALANCE) - 1) * 100
    
    report_text = f"""
=====================================
ОТЧЕТ О БЭКТЕСТЕ AI TRADER
=====================================
Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Чекпоинт: {Path(CHECKPOINT).parent.name}/{Path(CHECKPOINT).name}
Период: {test_dates[0].date()} — {test_dates[-1].date()}

ФИНАНСОВЫЕ РЕЗУЛЬТАТЫ:
Начальный баланс: {INITIAL_BALANCE:,.2f} ₽
Финальный баланс:  {final_balance:,.2f} ₽
Чистая прибыль:    {(final_balance - INITIAL_BALANCE):,.2f} ₽
Доходность:        {profit_pct:+.2f}%
=====================================
"""
    with open(OUTPUT_DIR / "backtest_report.txt", "a", encoding="utf-8") as f:
        f.write(report_text + "\n")
    
    # Визуализация (оставил компактной для чистоты)
    plt.figure(figsize=(15, 12))
    plt.subplot(2, 1, 1)
    plt.plot(res_df['date'], res_df['balance'], color='#2ecc71', linewidth=2)
    plt.axhline(y=INITIAL_BALANCE, color='#e74c3c', linestyle='--')
    plt.title(f"Кривая капитала: {final_balance:,.2f} ₽ ({profit_pct:+.2f}%)", fontsize=14)
    plt.ylabel("Баланс")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    weights_matrix = np.array(res_df['weights'].tolist())
    plt.stackplot(res_df['date'], weights_matrix.T, alpha=0.8)
    plt.title("Динамика распределения активов", fontsize=14)
    plt.ylabel("Доля весов")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "evaluation_report.png")
    plt.close() # Закрываем график, чтобы не мусорить в консоли
    print("✅ Бэктест успешно завершен!")

if __name__ == "__main__":
    run_evaluation()