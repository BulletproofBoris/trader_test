import os
import json
from pathlib import Path
import re

def generate_specs():
    base_path = Path("data/processed")
    
    if not base_path.exists():
        print(f"❌ Базовая директория {base_path} не найдена.")
        return

    dataset_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    
    if not dataset_dirs:
        print("❌ Конфигурации данных не найдены.")
        return

    models_processed = 0

    for target_config in dataset_dirs:
        print(f"\n{'='*40}")
        print(f"🔍 Сканирование конфигурации: {target_config.name}")
        print(f"{'='*40}")

        folds = sorted([d for d in target_config.glob("fold_*") if d.is_dir()])

        for fold in folds:
            # 1. Проверяем признаки
            features_file = fold / "artifacts" / "features_selected.json"
            if not features_file.exists():
                print(f"  ⚠️ [Пропуск {fold.name}]: Нет файла {features_file.name}")
                continue

            with open(features_file, 'r', encoding='utf-8') as f:
                feature_list = json.load(f).get("feature_order", [])
                
            if not feature_list:
                print(f"  ⚠️ [Пропуск {fold.name}]: Список feature_order пуст")
                continue

            # 2. Проверяем модели
            models_dir = fold / "models"
            if not models_dir.exists():
                print(f"  ⚠️ [Пропуск {fold.name}]: Папка models/ не существует")
                continue
                
            keras_models = list(models_dir.glob("*.keras"))
            if not keras_models:
                print(f"  ⚠️ [Пропуск {fold.name}]: Не найдены файлы .keras")
                continue

            for model_file in keras_models:
                model_name = model_file.stem
                original_json_path = models_dir / f"{model_name}.json"
                
                val_acc = 0.0
                val_loss = 0.0
                
                if original_json_path.exists():
                    try:
                        with open(original_json_path, 'r', encoding='utf-8') as f:
                            orig_data = json.load(f)
                            val_acc = orig_data.get("metrics", {}).get("val_acc", 0.0)
                            val_loss = orig_data.get("metrics", {}).get("val_loss", 0.0)
                    except Exception:
                        pass
                else:
                    match = re.search(r'_acc_([\d\.]+)_', model_name)
                    if match:
                        val_acc = float(match.group(1))

                parts = target_config.name.split('_')
                try:
                    timeframe, lookback, horizon = parts[2], int(parts[3]), int(parts[4])
                except (IndexError, ValueError):
                    timeframe, lookback, horizon = "1d", 60, 10

                # 3. Собираем Metadata
                metadata = {
                    "model_name": model_name,
                    "fold": fold.name,
                    "version": "2.0",
                    "framework": "tensorflow",
                    "input_shape": [lookback, len(feature_list)], 
                    "features": feature_list,
                    "metrics": {
                        "val_acc": val_acc,
                        "val_loss": val_loss
                    },
                    "config": {
                        "timeframe": timeframe,
                        "lookback": lookback,
                        "horizon": horizon
                    }
                }

                meta_path = models_dir / f"{model_name}_metadata.json"
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=4, ensure_ascii=False)
                    
                # ОБНОВЛЕННЫЙ ВЫВОД (теперь с Loss)
                print(f"  ✅ {model_name} | Acc: {val_acc:5.2f}% | Loss: {val_loss:.4f} | Фичей: {len(feature_list)}")
                models_processed += 1

    print(f"\n🎉 Готово! Создано/обновлено спецификаций: {models_processed}")

if __name__ == "__main__":
    generate_specs()