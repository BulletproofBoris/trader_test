import pandas as pd
from pathlib import Path
import sys
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent

def check_data_quality(timeframe):
    DATA_DIR = BASE_DIR / "data" / "raw" / timeframe
    print(f"🔍 Аудит сырых данных в {DATA_DIR.relative_to(BASE_DIR)}...\n")
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("❌ CSV файлы не найдены.")
        return

    report = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            if df.empty or 'Date' not in df.columns: continue
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            nans = df.isna().sum().sum()
            dates = pd.Series(df['Date'].unique())
            max_gap = int(dates.diff().dt.days.max()) if len(dates) > 1 else 0
            zero_vol = (df['Volume'] == 0).sum() if 'Volume' in df.columns else 0
            
            report.append({
                'Инструмент': file_path.name.replace('.csv', ''),
                'Дней': len(df),
                'NaN': nans,
                'Макс. Гэп': max_gap,
                'Нулевой объем': zero_vol
            })
        except Exception as e:
            print(f"⚠️ Ошибка {file_path.name}: {e}")

    report_df = pd.DataFrame(report).sort_values(by=['NaN', 'Макс. Гэп'], ascending=[False, False])
    pd.set_option('display.max_rows', None)
    print(report_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeframe', type=str, default="1d")
    args = parser.parse_args()
    check_data_quality(args.timeframe)