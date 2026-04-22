import os
import json
from pathlib import Path

# Настройки
BASE_DIR = Path("data/processed")
MODELS_TO_KEEP = 3  # Сколько лучших моделей оставить в каждом фолде
METRIC_TO_SORT = "val_loss"  # Главный критерий отбора
REVERSE_SORT = False # False для loss (ищем минимум), True если бы сортировали по accuracy

def clean_models():
    print(f"🧹 Запуск универсальной уборки моделей (Оставляем Топ-{MODELS_TO_KEEP} по {METRIC_TO_SORT})...")
    
    if not BASE_DIR.exists():
        print(f"❌ Базовая директория {BASE_DIR} не найдена!")
        return

    # Рекурсивно ищем все папки fold_* внутри data/processed
    folds = sorted([d for d in BASE_DIR.rglob("fold_*") if d.is_dir()])
    
    total_deleted = 0
    total_saved = 0
    folds_processed = 0
    
    for fold_dir in folds:
        models_dir = fold_dir / "models"
        if not models_dir.exists():
            continue
            
        # dataset_name поможет нам понимать, в какой именно песочнице мы сейчас убираемся
        dataset_name = fold_dir.parent.name
        print(f"\n📂 [{dataset_name}] Проверка фолда: {fold_dir.name}")
        
        # Собираем все мета-файлы
        meta_files = list(models_dir.glob("*_meta.json"))
        models_data = []
        
        for meta_file in meta_files:
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                keras_filename = meta_file.name.replace("_meta.json", ".keras")
                keras_file = models_dir / keras_filename
                
                if keras_file.exists():
                    models_data.append({
                        "meta_path": meta_file,
                        "keras_path": keras_file,
                        "val_loss": data.get("val_loss", float('inf')),
                        "accuracy": data.get("best_val_accuracy", 0.0),
                        "run_id": data.get("run_id", "unknown")
                    })
            except Exception as e:
                print(f"  ⚠️ Ошибка чтения {meta_file.name}: {e}")
                
        if not models_data:
            print("  Пусто. Нет пар (.keras + .json).")
            continue
            
        folds_processed += 1
        
        # Сортируем модели
        models_data.sort(key=lambda x: x[METRIC_TO_SORT], reverse=REVERSE_SORT)
        
        # Определяем, кого оставляем, а кого удаляем
        keepers = models_data[:MODELS_TO_KEEP]
        trash = models_data[MODELS_TO_KEEP:]
        
        print(f"  🏆 Оставляем:")
        for i, m in enumerate(keepers):
            print(f"     {i+1}. Run {m['run_id']} | Loss: {m['val_loss']:.4f} | Acc: {m['accuracy']*100:.2f}%")
            total_saved += 1
            
        if trash:
            print(f"  🗑️  Удаляем {len(trash)} файлов...")
            for m in trash:
                try:
                    m['keras_path'].unlink()
                    m['meta_path'].unlink()
                    total_deleted += 1
                except Exception as e:
                    print(f"     ❌ Ошибка удаления {m['keras_path'].name}: {e}")
        else:
            print("  ✨ Удалять нечего, количество моделей в норме.")

    print("\n" + "="*50)
    print(f"🏁 Уборка завершена!")
    print(f"Обработано фолдов:         {folds_processed}")
    print(f"Сохранено элитных моделей: {total_saved}")
    print(f"Удалено мусорных моделей:  {total_deleted} (освобождено место на диске)")
    print("="*50)

if __name__ == "__main__":
    clean_models()