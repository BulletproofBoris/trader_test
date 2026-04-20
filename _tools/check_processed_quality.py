import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent

def check_processed_quality(timeframe):
    PROCESSED_DIR = BASE_DIR / "data" / "processed" / timeframe
    print(f"🔍 Аудит сгенерированных фичей в {PROCESSED_DIR.relative_to(BASE_DIR)}...\n")
    
    csv_files = list(PROCESSED_DIR.glob("*_processed.csv"))
    if not csv_files:
        print("❌ Файлы не найдены."); return
        
    report = []
    base_columns = None
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            if base_columns is None: base_columns = set(df.columns)
            
            col_match = (set(df.columns) == base_columns)
            nans = df.isna().sum().sum()
            numeric_df = df.select_dtypes(include=[np.number])
            infs = np.isinf(numeric_df).sum().sum()
            out_of_bounds = ((numeric_df < 0) | (numeric_df > 1)).sum().sum() if not numeric_df.empty else 0
            constant_features = (numeric_df.nunique() <= 1).sum()
            
            report.append({
                'Файл': file_path.name.replace('_processed.csv', ''),
                'Строк': len(df),
                'Структура ОК?': '✅' if col_match else '❌',
                'NaN / Inf': f"{nans} / {infs}",
                'Вне [0,1]': out_of_bounds,
                'Мертвые фичи': constant_features
            })
        except Exception as e:
            print(f"⚠️ Ошибка {file_path.name}: {e}")

    report_df = pd.DataFrame(report).sort_values(by=['Вне [0,1]', 'Мертвые фичи'], ascending=[False, False])
    pd.set_option('display.max_rows', None)
    print(report_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeframe', type=str, default="1d")
    args = parser.parse_args()
    check_processed_quality(args.timeframe)