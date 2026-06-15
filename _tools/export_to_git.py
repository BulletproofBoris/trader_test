import os
import json
import shutil
from pathlib import Path

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    RL_DIR = BASE_DIR / "data" / "processed" / "2000_2026_1d" / "rl_env"
    EXPERIMENT_DIR = RL_DIR / "ray_results" / "pbt_trading_bot"
    GIT_EXPORT_DIR = RL_DIR / "champions"

    if not EXPERIMENT_DIR.exists():
        print(f"❌ Директория с результатами не найдена: {EXPERIMENT_DIR}")
        return

    print("🔍 Поиск лучшего чекпоинта через прямой разбор метаданных Ray Tune...")
    
    best_score = -float('inf')
    best_checkpoint_dir = None
    best_metrics = {}

    # Напрямую сканируем папки триалов PPO_*
    for trial_dir in EXPERIMENT_DIR.glob("PPO_*"):
        if not trial_dir.is_dir(): 
            continue
        
        result_file = trial_dir / "result.json"
        if not result_file.exists(): 
            continue
        
        # Читаем последнюю строчку файла результатов (финальное состояние воркера)
        last_result = None
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        last_result = json.loads(line)
        except Exception:
            continue
            
        if not last_result: 
            continue
            
        # Извлекаем метрику доходности
        score = last_result.get('env_runners', {}).get('episode_return_mean', -float('inf'))
        
        # Находим глобального лидера всей популяции
        if score > best_score:
            checkpoints = sorted(list(trial_dir.glob("checkpoint_*")), key=lambda x: x.name)
            if checkpoints:
                best_score = score
                best_checkpoint_dir = checkpoints[-1] # Берем самый свежий чекпоинт лидера
                best_metrics = last_result

    if not best_checkpoint_dir:
        print("⚠️ Валидные чекпоинты не найдены. Возможно, обучение завершилось до первого сохранения.")
        return

    print("\n" + "="*60)
    print(f"🏆 НАЙДЕН АБСОЛЮТНЫЙ ЧЕМПИОН ПОПУЛЯЦИИ!")
    print(f"  Trial ID: {best_checkpoint_dir.parent.name}")
    print(f"  Checkpoint: {best_checkpoint_dir.name}")
    print(f"  Средняя альфа-доходность (Return Mean): {best_score:.2f} б.п.")
    print("="*60)
    
    # Регенерация папки чемпионов для Git
    if GIT_EXPORT_DIR.exists():
        shutil.rmtree(GIT_EXPORT_DIR)
    os.makedirs(GIT_EXPORT_DIR, exist_ok=True)
    
    # Копируем мозг лучшего агента
    shutil.copytree(best_checkpoint_dir, GIT_EXPORT_DIR / "best_model")
    print(f"✅ Чекпоинт скопирован в: {GIT_EXPORT_DIR / 'best_model'}")
    
    # Экспортируем файлы прогресса и бэктестов, если они есть
    for src_file, dst_name in [
        (RL_DIR / "training_progress.csv", "training_progress.csv"),
        (BASE_DIR / "evaluation_report.html", "evaluation_report.html")
    ]:
        if src_file.exists():
            shutil.copy(src_file, GIT_EXPORT_DIR / dst_name)
            print(f"✅ Файл {dst_name} успешно перенесен в champions.")
            
    print("\n🎉 Автономный экспорт успешно завершен!")

if __name__ == "__main__":
    main()