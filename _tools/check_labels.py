import pandas as pd
from pathlib import Path
import sys
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent

def check_labels_quality(exp_name):
    LABELED_DIR = BASE_DIR / "experiments" / exp_name / "labels"
    print(f"🔍 Аудит разметки для {exp_name}...\n")
    
    csv_files = list(LABELED_DIR.glob("*_processed.csv"))
    if not csv_files: print("❌ Файлы не найдены."); return
        
    report, total_c0, total_c1, total_c2, total_rows = [], 0, 0, 0, 0
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, usecols=['label'])
            counts = df['label'].value_counts()
            c0, c1, c2 = counts.get(0.0, 0), counts.get(1.0, 0), counts.get(2.0, 0)
            n_rows = len(df)
            total_rows += n_rows; total_c0 += c0; total_c1 += c1; total_c2 += c2
            
            report.append({
                'Тикер': file_path.name.replace('_processed.csv', ''),
                '0 (Флэт)': f"{(c0/n_rows)*100:.1f}%",
                '1 (TP)': f"{(c1/n_rows)*100:.1f}%",
                '2 (SL)': f"{(c2/n_rows)*100:.1f}%"
            })
        except Exception as e:
            pass

    pd.set_option('display.max_rows', None)
    print(pd.DataFrame(report).to_string(index=False))
    
    print("\n" + "=" * 60)
    print(f"📈 ГЛОБАЛЬНЫЙ БАЛАНС")
    print(f"🕒 Флэт: {(total_c0/total_rows)*100:.1f}% | 🚀 TP: {(total_c1/total_rows)*100:.1f}% | 🩸 SL: {(total_c2/total_rows)*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    args = parser.parse_args()
    check_labels_quality(args.exp_name)