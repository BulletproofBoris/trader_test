import json
import os
from pathlib import Path
import shutil

def clean_old_checkpoints():
    RL_DIR = Path("data/processed/2000_2026_1d/rl_env")
    HEALTHY_JSON = RL_DIR / "healthy_checkpoints.json"
    EXPERIMENT_DIR = RL_DIR / "ray_results" / "pbt_trading_bot"

    with open(HEALTHY_JSON, 'r') as f:
        healthy_paths = set(json.load(f))

    count = 0
    for ckpt_dir in EXPERIMENT_DIR.glob("**/checkpoint_*"):
        if str(ckpt_dir.absolute()) not in healthy_paths:
            shutil.rmtree(ckpt_dir)
            count += 1
    
    print(f"🧹 Удалено {count} устаревших чекпоинтов. Оставлены только отобранные 'здоровые'.")

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    RL_DIR = BASE_DIR / "data" / "processed" / "2000_2026_1d" / "rl_env"
    EXPERIMENT_DIR = RL_DIR / "ray_results" / "pbt_trading_bot"
    OUTPUT_JSON = RL_DIR / "healthy_checkpoints.json"

    if not EXPERIMENT_DIR.exists():
        print(f"❌ Директория {EXPERIMENT_DIR} не найдена!")
        return

    print("🔍 Поиск лучших чекпоинтов строго до ПЕРВОГО перехода на 3 фазу...")
    
    healthy_checkpoints = []

    for trial_dir in EXPERIMENT_DIR.glob("PPO_*"):
        if not trial_dir.is_dir():
            continue
            
        result_file = trial_dir / "result.json"
        if not result_file.exists():
            continue

        # Шаг 1: Ищем ПЕРВОЕ касание 3-й фазы
        first_phase3_iter = float('inf')
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    record = json.loads(line)
                    phase = record.get('env_runners', {}).get('custom_metrics', {}).get('task_phase_max', 1.0)
                    iteration = record.get('training_iteration', 0)
                    
                    # Как только впервые коснулись 3 фазы, запоминаем итерацию и перестаем проверять дальше
                    if phase >= 3.0:
                        if iteration < first_phase3_iter:
                            first_phase3_iter = iteration
                            
        except Exception as e:
            print(f"⚠️ Ошибка чтения лога {result_file.name}: {e}")
            continue

        if first_phase3_iter == float('inf'):
            print(f"⚠️ В триале {trial_dir.name[:15]} 3-я фаза вообще не начиналась.")
            continue

        # Шаг 2: Ищем чекпоинт, который был создан строго до этого момента
        best_ckpt = None
        best_ckpt_iter = -1
        
        for ckpt_dir in trial_dir.glob("checkpoint_*"):
            if not ckpt_dir.is_dir(): 
                continue
            
            try:
                ckpt_iter = int(ckpt_dir.name.split('_')[1])
            except ValueError:
                continue
            
            # Чекпоинт должен быть ДО первого появления 3 фазы
            if best_ckpt_iter < ckpt_iter < first_phase3_iter:
                best_ckpt_iter = ckpt_iter
                best_ckpt = ckpt_dir

        if best_ckpt:
            healthy_checkpoints.append(str(best_ckpt.absolute()))
            print(f"✅ {trial_dir.name[:15]}... | 3-я фаза: Итер {first_phase3_iter} | Взят чекпоинт: {best_ckpt.name}")
        else:
            print(f"❌ {trial_dir.name[:15]}... | 3-я фаза на {first_phase3_iter}, но чекпоинтов ДО нее нет!")

    if healthy_checkpoints:
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(healthy_checkpoints, f, indent=4)
        print(f"\n💾 Успешно сохранено {len(healthy_checkpoints)} чекпоинтов в {OUTPUT_JSON.name}")
    else:
        print("❌ Не найдено ни одного подходящего чекпоинта.")

if __name__ == "__main__":
    main()
    clean_old_checkpoints()