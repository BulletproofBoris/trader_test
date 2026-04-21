import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import logging
import torch # Нам понадобится чистый PyTorch для работы с модулем

# --- ГЛУШИМ СПАМ ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
warnings.filterwarnings("ignore")

# --- ДОБАВЛЯЕМ КОРЕНЬ ПРОЕКТА В ПУТИ PYTHON ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from _tools.rl_env import TradingEnv
from _tools.audit_models import audit_models

# Импортируем новый API для работы с модулями
from ray.rllib.core.rl_module.rl_module import RLModule

RL_DIR = BASE_DIR / "data" / "processed" / "2000_2026_1d" / "rl_env"
REPORT_FILE = RL_DIR / "tournament_2025_2026_results.csv"

def run_tournament():
    # Нам больше не нужен тяжелый кластер Ray для эмуляции
    # ray.init() мы даже не вызываем!
    
    print("Собираем топ претендентов...")
    top_models = audit_models()
    
    if top_models is None or top_models.empty:
        print("\n❌ Турнир отменен: нет подходящих моделей на диске.")
        return
        
    results = []
    print(f"\n🏆 Запускаем турнир 2025-2026 годов для {len(top_models)} моделей...\n")

    for i, (_, row) in enumerate(top_models.iterrows()):
        trial_id = row['Trial_ID']
        iteration = row['Iteration']
        
        # В новом API веса лежат в папке learner_group/learner/rl_module/default_policy
        cp_path = os.path.join(row['Path'], f"checkpoint_{str(iteration).zfill(6)}")
        module_path = os.path.join(cp_path, "learner_group", "learner", "rl_module", "default_policy")
        
        print(f"[{i+1}/{len(top_models)}] Загрузка нейросети {trial_id}_v{iteration}...")
        
        try:
            # 1. ЗАГРУЖАЕМ ЧИСТЫЙ МОЗГ АГЕНТА (Без Ray Workers)
            if not os.path.exists(module_path):
                print(f"   ⚠️ Файлы модуля не найдены по пути: {module_path}. Пропускаем.")
                continue
                
            rl_module = RLModule.from_checkpoint(module_path)
            
            # 2. ИНИЦИАЛИЗИРУЕМ СРЕДУ 2025-2026
            env_config = {
                "data_path": str(RL_DIR / "environment_data.parquet"),
                "split_mode": "2025_2026",
                "initial_balance": 100000.0,
                "commission": 0.0003
            }
            env = TradingEnv(env_config)
            
            if env.total_steps <= 0:
                 print("   ⚠️ Ошибка: В среде нет данных за 2025-2026. Симуляция отменена.")
                 break # Если данных нет, останавливаем весь турнир
                 
            # 3. БОЕВАЯ ЭМУЛЯЦИЯ
            obs, info = env.reset()
            done = False
            
            while not done:
                # Преобразуем numpy массив в тензор PyTorch (добавляем batch_size=1)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                
                # Прогоняем через нейросеть
                with torch.no_grad():
                    # forward_inference используется для применения в бою
                    action_logits = rl_module.forward_inference({"obs": obs_tensor})
                    # Берем индекс максимального значения (action = 0, 1 или 2)
                    action = torch.argmax(action_logits["action_dist_inputs"]).item()
                
                # Делаем шаг в среде
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            final_profit_pct = ((info['balance'] - 100000.0) / 100000.0) * 100
            
            results.append({
                "Model_ID": f"{trial_id}_v{iteration}",
                "Profit_25_26_%": round(final_profit_pct, 2),
                "Final_Balance": round(info['balance'], 2),
                "Train_History_%": row['Train_Ret'],
                "Test_History_%": row['Test_Ret']
            })
            
            print(f"   ✅ Завершено. Профит: {final_profit_pct:.2f}%")
            
        except Exception as e:
            print(f"   ❌ Ошибка при тесте: {e}")

    if not results:
        print("\n❌ Ни одна модель не смогла пройти симуляцию.")
        return

    # Сохраняем финальный отчет
    report_df = pd.DataFrame(results).sort_values("Profit_25_26_%", ascending=False)
    report_df.to_csv(REPORT_FILE, index=False)
    
    print("\n" + "="*80)
    print(f"🏁 ТУРНИР 2025-2026 ГОДОВ ЗАВЕРШЕН! Результаты в: {REPORT_FILE}")
    print("="*80)
    format_str = "{:<25} | {:<15} | {:<15} | {:<15}"
    print(format_str.format("Model ID", "Profit 25-26", "Train Hist", "Test Hist"))
    print("-" * 80)
    for _, row in report_df.head(10).iterrows():
        print(format_str.format(
            row['Model_ID'], 
            f"{row['Profit_25_26_%']}%", 
            f"{row['Train_History_%']}%", 
            f"{row['Test_History_%']}%"
        ))

if __name__ == "__main__":
    run_tournament()