import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    # 1. Настраиваем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Оркестратор массового обучения LSTM (Walk-Forward)")
    parser.add_argument("--dataset_dir", type=str, default="data/processed/2000_2026_1d_6_1", help="Путь к корневой папке датасета")
    parser.add_argument("--runs", type=int, default=100, help="Количество прогонов (runs) для каждого фолда")
    # Удален параметр --batch_size, так как теперь он адаптивный!
    parser.add_argument("--epochs", type=int, default=100, help="Количество эпох обучения")
    parser.add_argument("--l2_reg", type=str, default="1e-3", help="Коэффициент L2 регуляризации")
    parser.add_argument("--lr", type=str, default="8e-2", help="Стартовый Learning Rate")
    parser.add_argument("--start_fold", type=str, default="fold_2010", help="Имя фолда, с которого начать (например, fold_2018)")
    
    # НОВЫЕ ПАРАМЕТРЫ ЭЛАСТИЧНОГО ТЕРПЕНИЯ
    parser.add_argument("--bonus_ratio", type=float, default=0.1, help="Доля от микро-лимита, добавляемая за рекорд")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Минимальное улучшение Loss для получения бонуса")
    
    # Флаги-переключатели
    parser.add_argument("--append", action="store_true", help="Дообучать поверх существующих моделей")
    parser.add_argument("--force", action="store_true", help="Очистить папку с моделями перед стартом")

    args = parser.parse_args()

    DATASET_DIR = Path(args.dataset_dir)
    print("🚀 Запуск массового обучения моделей (Walk-Forward)...")
    print(f"📁 Датасет: {DATASET_DIR}")
    print(f"⚙️  Настройки: {args.runs} runs, {args.epochs} epochs (Батч вычисляется динамически!)")
    print(f"🧠 Эластичность: Bonus Ratio = {args.bonus_ratio}, Min Delta = {args.min_delta}")

    if not DATASET_DIR.exists():
        print(f"❌ Ошибка: Директория {DATASET_DIR} не найдена!")
        sys.exit(1)

    folds = sorted([d for d in DATASET_DIR.glob("fold_*") if d.is_dir()])

    if args.start_fold:
        fold_names = [f.name for f in folds]
        if args.start_fold in fold_names:
            start_idx = fold_names.index(args.start_fold)
            folds = folds[start_idx:] 
            print(f"⏭️ Пропускаем завершенные фолды. Начинаем строго с: {args.start_fold}")
        else:
            print(f"❌ Ошибка: Фолд {args.start_fold} не найден в директории!")
            sys.exit(1)

    if not folds:
        print("⚠️ Фолды для обучения не найдены.")
        sys.exit(0)

    try:
        for fold_dir in folds:
            fold_name = fold_dir.name
            
            print("\n" + "="*60)
            print(f"🔥 Обучение нейросети для: {fold_name}")
            print("="*60)
            
            # Формируем базовую команду вызова
            # Используем прямой вызов файла, чтобы точно избежать проблем с путями
            script_path = str(Path("_tools") / "train_ltsm_model.py")
            
            cmd = [
                "python", script_path,
                "--dataset_dir", str(DATASET_DIR),
                "--fold", fold_name,
                "--runs", str(args.runs),
                "--epochs", str(args.epochs),
                "--l2_reg", args.l2_reg,
                "--lr", args.lr,
                "--bonus_ratio", str(args.bonus_ratio),
                "--min_delta", str(args.min_delta)
            ]
            
            if args.append:
                cmd.append("--append")
            if args.force:
                cmd.append("--force")
            
            process = subprocess.Popen(cmd)
            process.wait()
            
            if process.returncode == 0:
                print(f"✅ [{fold_name}] Завершен успешно!")
            else:
                print(f"⚠️ [{fold_name}] Завершился с кодом ошибки: {process.returncode}")

    except KeyboardInterrupt:
        print("\n🛑 Остановка пайплайна пользователем!")
        if 'process' in locals():
            process.terminate()
            print("⏳ Завершаем текущий фолд...")
    except Exception as e:
        print(f"❌ Ошибка пайплайна: {e}")

    print("🎉 Процесс полностью остановлен.")

if __name__ == "__main__":
    main()