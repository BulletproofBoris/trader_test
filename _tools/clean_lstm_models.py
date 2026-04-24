import sys
import re
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Глобальная очистка мусорных и временных файлов моделей")
    parser.add_argument("--base_dir", type=str, default="data/processed", help="Путь к базовой папке со всеми датасетами")
    parser.add_argument("--keep", type=int, default=3, help="Сколько лучших моделей оставить в каждом фолде")
    
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        print(f"❌ Ошибка: Базовая директория {base_dir} не найдена!")
        sys.exit(1)

    print(f"🧹 Запуск ГЛОБАЛЬНОЙ уборки моделей в: {base_dir} (Оставляем Топ-{args.keep})...")
    
    model_pattern = re.compile(r"loss_([0-9]+\.[0-9]+).*?\.keras")

    dataset_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    if not dataset_dirs:
        print(f"⚠️ В папке {base_dir} не найдено датасетов.")
        sys.exit(0)

    grand_total_models_deleted = 0
    grand_total_temp_deleted = 0

    for dataset_dir in dataset_dirs:
        print("\n" + "#"*80)
        print(f"🚀 ОБРАБОТКА ДАТАСЕТА: {dataset_dir.name}")
        print("#"*80)

        folds = sorted([d for d in dataset_dir.glob("fold_*") if d.is_dir()])
        
        if not folds:
            print(f"   ⚠️ Фолдов не найдено, пропускаем...")
            continue

        dataset_models_deleted = 0
        dataset_temp_deleted = 0

        for fold_dir in folds:
            models_dir = fold_dir / "models"
            if not models_dir.exists():
                continue

            print(f"\n📂 Проверка: {fold_dir.name}")
            
            temp_files_found = []
            
            # 1. ТОЧЕЧНЫЙ ПОИСК ВРЕМЕННЫХ ФАЙЛОВ И МУСОРА (БЕЗ УДАЛЕНИЯ МЕТАДАННЫХ)
            for file_path in models_dir.iterdir():
                is_trash = False
                
                if file_path.is_file():
                    name = file_path.name
                    # Удаляем только явно известный системный мусор и сырые веса
                    if name.startswith("temp_"):
                        is_trash = True
                    elif name.endswith(".h5") or name.endswith(".weights.h5"):
                        is_trash = True
                    elif file_path.suffix in [".tmp", ".temp", ".part", ".index", ".data-00000-of-00001"]:
                        is_trash = True
                    elif name == "checkpoint":
                        is_trash = True
                
                # Ищем папки-призраки вида run.keras.tmp
                elif file_path.is_dir() and file_path.name.endswith(".tmp"):
                    is_trash = True
                
                if is_trash:
                    temp_files_found.append(file_path)

            if temp_files_found:
                for tf_path in temp_files_found:
                    try:
                        if tf_path.is_file():
                            tf_path.unlink()
                        else:
                            shutil.rmtree(tf_path)
                        dataset_temp_deleted += 1
                        grand_total_temp_deleted += 1
                    except Exception as e:
                        print(f"   ⚠️ Ошибка удаления временного файла {tf_path.name}: {e}")
                print(f"   🧹 Удалено временных файлов/чекпоинтов: {len(temp_files_found)}")

            # 2. ФИЛЬТРАЦИЯ И УДАЛЕНИЕ ХУДШИХ .keras МОДЕЛЕЙ
            keras_files = list(models_dir.glob("*.keras"))
            valid_models = []

            for m_file in keras_files:
                match = model_pattern.search(m_file.name)
                if match:
                    loss = float(match.group(1))
                    valid_models.append({"path": m_file, "loss": loss})

            valid_models.sort(key=lambda x: x["loss"])

            if not valid_models:
                print("   ⚠️ Целых моделей .keras не найдено.")
                continue

            elites = valid_models[:args.keep]
            trash = valid_models[args.keep:]

            print("   🏆 Оставляем:")
            for i, elite in enumerate(elites, 1):
                print(f"      {i}. {elite['path'].name}")

            if trash:
                for bad_model in trash:
                    try:
                        bad_model["path"].unlink()
                        dataset_models_deleted += 1
                        grand_total_models_deleted += 1
                    except Exception as e:
                        print(f"   ⚠️ Ошибка удаления {bad_model['path'].name}: {e}")
                print(f"   🗑️  Удаляем {len(trash)} файлов...")
            else:
                print("   ✨ Удалять нечего, количество моделей в норме.")

        print(f"\n📊 Итоги по датасету {dataset_dir.name}: удалено {dataset_models_deleted} моделей, {dataset_temp_deleted} временных файлов.")

    print("\n" + "="*80)
    print("🏁 ГЛОБАЛЬНАЯ УБОРКА ПОЛНОСТЬЮ ЗАВЕРШЕНА!")
    print(f"Всего удалено мусорных моделей:  {grand_total_models_deleted}")
    print(f"Всего удалено временных файлов:  {grand_total_temp_deleted}")
    print("="*80)

if __name__ == "__main__":
    main()