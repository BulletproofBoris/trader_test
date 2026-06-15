import os
import shutil
from pathlib import Path
from ray import tune

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    RL_DIR = BASE_DIR / "data" / "processed" / "2000_2026_1d" / "rl_env"
    EXPERIMENT_DIR = RL_DIR / "ray_results" / "pbt_trading_bot"
    GIT_EXPORT_DIR = RL_DIR / "champions"

    if not EXPERIMENT_DIR.exists():
        print(f"❌ Директория с результатами обучения не найдена: {EXPERIMENT_DIR}")
        return

    print("🔍 Сканирование всех агентов в популяции для поиска лучшего чекпоинта...")
    try:
        tuner = tune.Tuner.restore(str(EXPERIMENT_DIR))
        result_grid = tuner.get_results()
        
        # Находим лучший результат по всем воркерам (кто заработал больше всего денег)
        best_result = result_grid.get_best_result(
            metric="env_runners/episode_return_mean", 
            mode="max"
        )
        
        best_checkpoint = best_result.checkpoint
        
        if best_checkpoint is None:
            print("⚠️ Чекпоинты не найдены. Вы прервали обучение до первого сохранения?")
            return
            
        checkpoint_path = best_checkpoint.path
        score = best_result.metrics.get("env_runners/episode_return_mean", 0)
        sharpe = best_result.metrics.get("custom_metrics", {}).get("sharpe_mean", 0)
        
        print(f"🏆 Найден ЛУЧШИЙ чекпоинт!")
        print(f"   Средняя награда (Доходность): {score:.2f}")
        print(f"   Коэффициент Шарпа: {sharpe:.2f}")
        print(f"   Путь: {checkpoint_path}")

        # Создаем папку для экспорта (чистим старую)
        if GIT_EXPORT_DIR.exists():
            shutil.rmtree(GIT_EXPORT_DIR)
        os.makedirs(GIT_EXPORT_DIR, exist_ok=True)
        
        # Копируем чекпоинт
        dest_checkpoint = GIT_EXPORT_DIR / "best_model"
        shutil.copytree(checkpoint_path, dest_checkpoint)
        print(f"✅ Чекпоинт скопирован в: {dest_checkpoint}")
        
        # Копируем логи
        csv_log = RL_DIR / "training_progress.csv"
        if csv_log.exists():
            shutil.copy(csv_log, GIT_EXPORT_DIR / "training_progress.csv")
            print("✅ CSV лог скопирован.")
            
        html_report = BASE_DIR / "evaluation_report.html"
        if html_report.exists():
            shutil.copy(html_report, GIT_EXPORT_DIR / "evaluation_report.html")
            print("✅ HTML отчет скопирован.")
            
        print("\n🎉 Экспорт успешно завершен! Теперь вы можете безопасно сделать git commit.")

    except Exception as e:
        print(f"❌ Произошла ошибка при экспорте: {e}")

if __name__ == "__main__":
    main()
