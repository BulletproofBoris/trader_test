import sys
import json
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Глобальная очистка моделей с подробным отчетом")
    parser.add_argument("--base_dir", type=str, default="data/processed", help="Путь к базовой папке")
    parser.add_argument("--keep", type=int, default=3, help="Сколько лучших моделей оставить")
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"❌ Ошибка: {base_dir} не найдена!")
        sys.exit(1)

    print(f"🧹 Запуск ГЛОБАЛЬНОЙ уборки в: {base_dir} (Оставляем Топ-{args.keep} по val_loss)...")
    
    dataset_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    grand_total_deleted = 0
    grand_total_temp = 0
    grand_total_kept = 0

    for dataset_dir in dataset_dirs:
        folds = sorted([d for d in dataset_dir.glob("fold_*") if d.is_dir()])
        if not folds: continue

        print("\n" + "#"*80)
        print(f"🚀 ДАТАСЕТ: {dataset_dir.name}")
        print("#"*80)

        for fold_dir in folds:
            models_dir = fold_dir / "models"
            if not models_dir.exists(): continue

            print(f"\n📂 Фолд: {fold_dir.name}")
            
            # 1. ЖЕСТКАЯ ОЧИСТКА ВРЕМЕННОГО МУСОРА
            temp_files = []
            for f in models_dir.iterdir():
                is_trash = False
                name = f.name
                if f.is_file():
                    if name.startswith("temp_") or name.endswith(".h5") or name.endswith(".weights.h5"):
                        is_trash = True
                    elif f.suffix in [".tmp", ".temp", ".part", ".index", ".data-00000-of-00001"]:
                        is_trash = True
                    elif name == "checkpoint":
                        is_trash = True
                elif f.is_dir() and name.endswith(".tmp"):
                    is_trash = True
                
                if is_trash:
                    temp_files.append(f)

            for tf_path in temp_files:
                try:
                    if tf_path.is_file(): tf_path.unlink()
                    else: shutil.rmtree(tf_path)
                    grand_total_temp += 1
                except Exception: pass
            
            if temp_files:
                print(f"   🧹 Удалено системного мусора: {len(temp_files)} файлов")

            # 2. АНАЛИЗ И ФИЛЬТРАЦИЯ МОДЕЛЕЙ (ЧЕРЕЗ JSON)
            keras_files = list(models_dir.glob("*.keras"))
            valid_models = []

            for m_file in keras_files:
                if m_file in temp_files: continue
                
                json_file = m_file.with_suffix(".json")
                val_loss = float('inf')
                val_acc = 0.0
                run_id = "?"
                
                # Читаем характеристики из JSON
                if json_file.exists():
                    try:
                        with open(json_file, 'r') as jf:
                            meta = json.load(jf)
                            # Аккуратно достаем метрики, если они есть
                            v_loss = meta.get("metrics", {}).get("val_loss")
                            val_loss = float(v_loss) if v_loss is not None else float('inf')
                            v_acc = meta.get("metrics", {}).get("val_acc")
                            val_acc = float(v_acc) if v_acc is not None else 0.0
                            run_id = meta.get("run_id", "?")
                    except Exception:
                        pass
                
                valid_models.append({
                    "path": m_file, 
                    "json": json_file,
                    "loss": val_loss,
                    "acc": val_acc,
                    "run": run_id
                })

            # Сортируем по Loss (от лучшего к худшему)
            valid_models.sort(key=lambda x: x["loss"])

            elites = valid_models[:args.keep]
            trash = valid_models[args.keep:]

            # Удаляем слабые модели И их JSON-паспорта
            if trash:
                for bad in trash:
                    try:
                        bad["path"].unlink()
                        if bad["json"].exists(): bad["json"].unlink()
                        grand_total_deleted += 1
                    except Exception: pass
                print(f"   🗑️  Списано слабых моделей: {len(trash)}")

            # Выводим оставшиеся (Элиту)
            if elites:
                print("   💎 Оставшиеся в строю (Топ по Loss):")
                for i, elite in enumerate(elites, 1):
                    loss_str = f"{elite['loss']:.4f}" if elite['loss'] != float('inf') else "N/A"
                    acc_str = f"{elite['acc']:.2f}%"
                    run_str = f"Run {elite['run']:>2}"
                    print(f"      {i}. {run_str} | Loss: {loss_str} | Acc: {acc_str} | Файл: {elite['path'].name}")
                grand_total_kept += len(elites)
            else:
                print("   ⚠️ Моделей не найдено.")

    # 3. ФИНАЛЬНЫЙ СВОДНЫЙ ОТЧЕТ
    print("\n" + "="*80)
    print("🏁 ИТОГОВЫЙ ОТЧЕТ ПО ФАБРИКЕ НЕЙРОСЕТЕЙ")
    print("="*80)
    print(f"🟢 Всего 'элитных' моделей готово к бою: {grand_total_kept}")
    print(f"🔴 Всего удалено слабых моделей:         {grand_total_deleted}")
    print(f"🧹 Всего очищено временных файлов:       {grand_total_temp}")
    print("="*80)

if __name__ == "__main__":
    main()