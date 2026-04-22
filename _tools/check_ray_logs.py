import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "data" / "processed" / "2000_2026_1d" / "rl_env" / "ray_results" / "pbt_trading_bot"

def check_logs():
    print(f"🔍 Проверка директории: {RESULTS_DIR}")
    
    if not RESULTS_DIR.exists():
        print("❌ Директория не найдена!")
        return

    # Ищем папки триалов (начинаются с PPO_)
    trial_dirs = [d for d in os.listdir(RESULTS_DIR) if d.startswith("PPO_") and os.path.isdir(RESULTS_DIR / d)]
    
    if not trial_dirs:
        print("❌ Папки триалов не найдены!")
        return
        
    print(f"✅ Найдено {len(trial_dirs)} папок триалов.")
    
    # Берем первую попавшуюся папку для анализа
    sample_dir = RESULTS_DIR / trial_dirs[0]
    print(f"\n📂 Анализ папки: {sample_dir.name}")
    
    # 1. Ищем чекпоинты
    checkpoints = [d for d in os.listdir(sample_dir) if "checkpoint_" in d]
    print(f"   Найдено чекпоинтов: {len(checkpoints)}")
    if checkpoints:
        print(f"   Примеры чекпоинтов: {checkpoints[:3]}")
        
    # 2. Анализируем result.json
    json_path = sample_dir / "result.json"
    if not json_path.exists():
        print("   ❌ Файл result.json не найден!")
        return
        
    print(f"   ✅ Файл result.json найден.")
    
    # Читаем последнюю строчку JSON'а
    last_line = None
    with open(json_path, 'r') as f:
        for line in f:
            last_line = line
            
    if last_line:
        data = json.loads(last_line)
        print("\n🔍 Структура последней записи в result.json:")
        print(f"   - Итерация: {data.get('training_iteration')}")
        
        # Пытаемся найти ключи с наградами
        env_runners = data.get('env_runners', {})
        print(f"   - env_runners -> episode_return_mean: {env_runners.get('episode_return_mean')}")
        
        eval_data = data.get('evaluation', {})
        eval_runners = eval_data.get('env_runners', {})
        print(f"   - evaluation -> env_runners -> episode_return_mean: {eval_runners.get('episode_return_mean')}")
        
        # Если старые ключи не сработали, выведем все ключи верхнего уровня
        if 'env_runners' not in data:
            print(f"   ⚠️ Доступные ключи: {list(data.keys())}")

if __name__ == "__main__":
    check_logs()