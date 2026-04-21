import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import torch

# --- ГЛУШИМ СПАМ ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
warnings.filterwarnings("ignore")

# --- ДОБАВЛЯЕМ КОРЕНЬ ПРОЕКТА В ПУТИ PYTHON ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from _tools.rl_env import TradingEnv
from ray.rllib.core.rl_module.rl_module import RLModule

RL_DIR = BASE_DIR / "data" / "processed" / "2000_2026_1d" / "rl_env"

def test_single_model():
    # Настройки для КОНКРЕТНОЙ модели
    trial_id = "7c64f_00003"
    iteration = 51
    
    print(f"\n🚀 ЗАПУСК ИНДИВИДУАЛЬНОГО ТЕСТА ДЛЯ 2026 ГОДА")
    print(f"Модель: {trial_id}_v{iteration}")
    print("="*60)
    
    # 1. Формируем путь к весам
    checkpoint_folder = f"checkpoint_{str(iteration).zfill(6)}"
    # Мы знаем примерный путь из предыдущих логов, но давай найдем его точно
    results_dir = RL_DIR / "ray_results" / "pbt_trading_bot"
    trial_path = None
    
    for item in os.listdir(results_dir):
        if trial_id in item and os.path.isdir(os.path.join(results_dir, item)):
            trial_path = os.path.join(results_dir, item)
            break
            
    if not trial_path:
        print(f"❌ Папка триала {trial_id} не найдена!")
        return
        
    module_path = os.path.join(trial_path, checkpoint_folder, "learner_group", "learner", "rl_module", "default_policy")
    
    if not os.path.exists(module_path):
        print(f"❌ Файлы нейросети не найдены по пути: {module_path}")
        return
        
    print("1. Загрузка весов нейросети...")
    rl_module = RLModule.from_checkpoint(module_path)
    
    # 2. Инициализируем среду ТОЛЬКО ДЛЯ 2026 ГОДА
    print("2. Подготовка среды для 2026 года...")
    env_config = {
        "data_path": str(RL_DIR / "environment_data.parquet"),
        "split_mode": "2026", # <--- ТОЛЬКО 2026 ГОД
        "initial_balance": 100000.0,
        "commission": 0.0003
    }
    env = TradingEnv(env_config)
    
    if env.total_steps <= 0:
         print("❌ Ошибка: В среде нет данных за 2026 год. Проверь parquet файл!")
         return
         
    print(f"   Найдено торговых дней: {env.total_steps}")
    
    # 3. Эмуляция торговли
    print("3. Запуск торговой эмуляции...")
    obs, info = env.reset()
    done = False
    
    # Собираем статистику для анализа
    history = []
    
    while not done:
        # Получаем дату текущего шага из DataFrame
        current_date = env.df.iloc[env.current_step]['datetime']
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_logits = rl_module.forward_inference({"obs": obs_tensor})
            action = torch.argmax(action_logits["action_dist_inputs"]).item()
        
        # Переводим action (0, 1, 2) в позицию (-1, 0, 1) для логирования
        position = action - 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        history.append({
            "date": current_date,
            "position": position,
            "balance": info['balance']
        })
        
    # 4. Вывод результатов
    final_balance = info['balance']
    profit_pct = ((final_balance - 100000.0) / 100000.0) * 100
    
    # Считаем количество сделок (переворотов)
    trades = 0
    for i in range(1, len(history)):
        if history[i]['position'] != history[i-1]['position']:
            trades += 1
            
    print("\n" + "="*60)
    print(f"🏆 РЕЗУЛЬТАТЫ ЭМУЛЯЦИИ: {trial_id}_v{iteration}")
    print("="*60)
    print(f"Стартовый баланс:  $100,000.00")
    print(f"Финальный баланс:  ${final_balance:,.2f}")
    print(f"Чистая прибыль:    {profit_pct:+.2f}%")
    print(f"Количество сделок: {trades}")
    print("="*60)
    
    # Выводим небольшую выдержку сделок
    print("\nПоследние 5 дней эмуляции:")
    for h in history[-5:]:
        pos_str = "LONG" if h['position'] == 1 else "SHORT" if h['position'] == -1 else "CASH"
        print(f"[{h['date'].strftime('%Y-%m-%d')}] Позиция: {pos_str:<5} | Баланс: ${h['balance']:,.2f}")

if __name__ == "__main__":
    test_single_model()