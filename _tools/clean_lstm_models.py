import sys
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Глобальная очистка моделей и логов с подробным отчетом")
    parser.add_argument("--base_dir", type=str, default="data/processed", help="Путь к базовой папке с моделями (или конкретному датасету)")
    parser.add_argument("--logs_dir", type=str, default=".", help="Папка, где лежат лог-файлы (по умолчанию текущая)")
    parser.add_argument("--keep", type=int, default=5, help="Сколько лучших моделей оставить")
    
    # 🚨 НОВЫЙ АРГУМЕНТ ДЛЯ ЦЕЛЕВОГО ФОЛДА 🚨
    parser.add_argument("--target_fold", type=str, default=None, help="Имя конкретного фолда для очистки (если не указано, ищутся все)")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    logs_dir = Path(args.logs_dir)
    
    if not base_dir.exists():
        print(f"❌ Ошибка: {base_dir} не найдена!")
        sys.exit(1)

    print(f"🧹 Запуск уборки...")
    print(f"   📂 Цель: {base_dir} (Оставляем Топ-{args.keep})")
    if args.target_fold:
         print(f"   🎯 Фолд: {args.target_fold}")
    
    # 1. ОЧИСТКА ЛОГОВ (worker_swarm_*.log) - Опционально, можно отключить
    deleted_logs = 0
    for log_file in logs_dir.glob("worker_swarm_*.log"):
        try:
            log_file.unlink()
            deleted_logs += 1
        except Exception as e:
            pass # Игнорируем ошибки удаления логов
    
    # 2. ОЧИСТКА МОДЕЛЕЙ
    
    # Определяем, передали ли нам папку датасета или базовую 'processed'
    if base_dir.name.startswith("20") and "1d" in base_dir.name:
         dataset_dirs = [base_dir]
    else:
         dataset_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and "1d" in d.name])
    
    grand_total_deleted = 0
    grand_total_kept = 0

    for dataset_dir in dataset_dirs:
        # Фильтруем фолды, если указан таргет
        if args.target_fold:
            folds = [dataset_dir / args.target_fold]
            if not folds[0].exists():
                print(f"⚠️ Фолд {args.target_fold} не найден в {dataset_dir.name}")
                continue
        else:
            folds = sorted([d for d in dataset_dir.glob("fold_*") if d.is_dir()])
            
        if not folds: continue

        print("\n" + "="*80)
        print(f"🚀 ДАТАСЕТ: {dataset_dir.name}")
        print("="*80)

        for fold_dir in folds:
            models_dir = fold_dir / "models"
            if not models_dir.exists(): continue

            print(f"\n📂 Фолд: {fold_dir.name}")
            
            # Временные файлы БОЛЬШЕ НЕ УДАЛЯЕМ, их чистят сами воркеры
            
            # Фильтрация моделей
            keras_files = list(models_dir.glob("*.keras"))
            valid_models = []

            for m_file in keras_files:
                json_file = m_file.with_suffix(".json")
                
                val_loss = float('inf')
                val_acc = 0.0
                run_id = "?"
                arch = "legacy" # По умолчанию, если файл старый
                
                if json_file.exists():
                    try:
                        import json as jmod
                        with open(json_file, 'r', encoding='utf-8') as jf:
                            meta = jmod.load(jf)
                            if "metrics" in meta:
                                val_loss = float(meta["metrics"].get("val_loss", float('inf')))
                                val_acc = float(meta["metrics"].get("val_acc", 0.0))
                            run_id = meta.get("run_id", "?")
                            # Извлекаем архитектуру (если ее нет, используем префикс файла)
                            arch = meta.get("arch", m_file.name.split('_')[0]) 
                    except Exception: pass
                
                valid_models.append({"path": m_file, "json": json_file, "loss": val_loss, "acc": val_acc, "run": run_id, "arch": arch})

            # ГРУППИРОВКА ПО АРХИТЕКТУРАМ
            models_by_arch = {}
            for m in valid_models:
                a = m["arch"]
                if a not in models_by_arch:
                    models_by_arch[a] = []
                models_by_arch[a].append(m)

            elites = []
            trash = []

            # Применяем квоту (--keep) к каждой архитектуре отдельно
            for a, models in models_by_arch.items():
                models.sort(key=lambda x: x["loss"])
                elites.extend(models[:args.keep])
                trash.extend(models[args.keep:])

            if trash:
                for bad in trash:
                    try:
                        bad["path"].unlink()
                        if bad["json"].exists(): bad["json"].unlink()
                        grand_total_deleted += 1
                    except Exception: pass
                print(f"   🗑️  Списано слабых моделей: {len(trash)}")

            if elites:
                print("   💎 Элита фолда (Топ по Loss):")
                # Сортируем вывод чисто для красоты (сначала по архитектуре, потом по Loss)
                elites.sort(key=lambda x: (x["arch"], x["loss"]))
                for elite in elites:
                    print(f"      [{elite['arch']}] Run: {elite['run']:<10} | Loss: {elite['loss']:.4f}")
                grand_total_kept += len(elites)

    print("\n" + "="*80)
    print("🏁 ИТОГОВЫЙ ОТЧЕТ")
    print("="*80)
    print(f"🟢 Моделей сохранено:      {grand_total_kept}")
    print(f"🔴 Удалено моделей:         {grand_total_deleted}")
    print("="*80)

if __name__ == "__main__":
    main()