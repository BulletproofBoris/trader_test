import pandas as pd
import numpy as np
from pathlib import Path
import json

def audit_dataset():
    base_path = Path("data/processed")
    dataset_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    if not dataset_dirs:
        print("❌ Данные не найдены.")
        return
    
    dataset_dir = dataset_dirs[-1]
    report_path = Path("data_audit_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as report:
        report.write(f"{'='*60}\nАУДИТ ДАТАСЕТА: {dataset_dir.name}\n{'='*60}\n")

        folds = sorted([d for d in dataset_dir.glob("fold_*") if d.is_dir()])
        
        for fold in folds:
            report.write(f"\n📁 ФОЛД: {fold.name}\n{'-'*30}\n")
            
            # Храним даты тренировки для проверки пересечений
            train_dates = set()
            
            for phase in ["train", "val"]:
                data_path = fold / "data" / phase / "ml_data.parquet"
                if not data_path.exists(): continue
                
                df = pd.read_parquet(data_path)
                
                if phase == "train":
                    train_dates = set(df['datetime'].unique())
                
                # 1. Баланс классов
                report.write(f"  🔹 {phase.upper()} ({len(df)} строк):\n")
                counts = df['label'].value_counts(normalize=True).sort_index()
                for lbl, pct in counts.items():
                    name = {1.0: "TP (+1)", -1.0: "SL (-1)", 0.0: "Hold (0)"}.get(lbl, "Unknown")
                    report.write(f"    {name:<10}: {pct*100:>6.2f}%\n")
                
                # 2. Качество данных
                nans = df.isna().sum().sum()
                report.write(f"    Пропуски: {nans} | Бесконечности: {np.isinf(df.select_dtypes(include=np.number).values).sum()}\n")
                
                # 3. Проверка на утечку
                if phase == "val":
                    val_dates = set(df['datetime'].unique())
                    overlap = train_dates.intersection(val_dates)
                    if overlap:
                        report.write(f"    ❌ УТЕЧКА БУДУЩЕГО: Найдено {len(overlap)} общих дней с Train!\n")
                    else:
                        report.write(f"    ✅ УТЕЧКА БУДУЩЕГО: Отсутствует (Чистый тест)\n")

                # 4. Проверка дубликатов
                dupes = df.duplicated(subset=['datetime', 'ticker']).sum()
                if dupes > 0:
                    report.write(f"    ⚠️ ВНИМАНИЕ: Найдено {dupes} дубликатов (ticker+date)!\n")

    print(f"✅ Аудит завершен: {report_path.absolute()}")

if __name__ == "__main__":
    audit_dataset()