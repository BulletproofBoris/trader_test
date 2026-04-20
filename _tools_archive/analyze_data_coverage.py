# _tools/analyze_data_coverage.py
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

# Импортируем пути и загрузчик из вашего проекта
from _core.paths import DATA_DIR, RESULTS_DIR, ensure_dirs
from _core.data_loader import load_data

# Гарантируем, что все папки существуют
ensure_dirs()
warnings.simplefilter('ignore', FutureWarning)

def get_date_range_for_ticker(ticker: str) -> dict | None:
    """
    Анализирует файл данных одного тикера и возвращает его временной диапазон.
    """
    df = load_data(ticker)
    
    if df is None or df.empty or 'datetime' not in df.columns:
        return {
            "ticker": ticker,
            "start_date": "N/A",
            "end_date": "N/A",
            "total_days": 0,
            "error": "File not found or empty"
        }
        
    try:
        start_date = df['datetime'].min().date()
        end_date = df['datetime'].max().date()
        total_days = len(df)
        
        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "error": None
        }
    except Exception as e:
        return {
            "ticker": ticker,
            "start_date": "N/A",
            "end_date": "N/A",
            "total_days": 0,
            "error": str(e)
        }

def main(args):
    print("--- 🔬 Анализ временного покрытия скачанных данных ---")

    # 1. Загружаем список тикеров
    universe_path = Path(args.universe_file)
    if not universe_path.exists():
        print(f"❌ Файл со списком акций не найден: {universe_path}"); return
        
    try:
        tickers = pd.read_csv(universe_path)['ticker'].tolist()
        print(f"Найдено {len(tickers)} тикеров в файле {args.universe_file}")
    except Exception as e:
        print(f"❌ Не удалось прочитать файл {universe_path}: {e}"); return

    # 2. Анализируем каждый тикер
    all_reports = []
    for ticker in tqdm(tickers, desc="Анализ файлов"):
        report = get_date_range_for_ticker(ticker)
        if report:
            all_reports.append(report)
            
    if not all_reports:
        print("⚠️ Не удалось проанализировать ни одного файла."); return

    # 3. Создаем отчет
    report_df = pd.DataFrame(all_reports)
    report_df = report_df[report_df['total_days'] > 0] # Исключаем тикеры без данных
    report_df.sort_values(by="start_date", ascending=False, inplace=True)
    
    report_dir = RESULTS_DIR / "reports"
    report_dir.mkdir(exist_ok=True)
    
    output_path_csv = report_dir / "data_coverage_report.csv"
    report_df.to_csv(output_path_csv, index=False)
    
    # --- 👇 НОВЫЙ БЛОК: ПОСТРОЕНИЕ ГИСТОГРАММЫ 👇 ---
    print("\nСоздание гистограммы распределения...")
    plt.figure(figsize=(14, 7))
    
    plt.hist(report_df['total_days'], bins=50, color='skyblue', edgecolor='black', alpha=0.8)
    
    plt.title('Распределение длины истории по акциям', fontsize=16)
    plt.xlabel('Количество торговых дней в истории')
    plt.ylabel('Количество акций')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Добавляем вертикальные линии для среднего и медианы
    mean_days = report_df['total_days'].mean()
    median_days = report_df['total_days'].median()
    plt.axvline(mean_days, color='red', linestyle='dashed', linewidth=2, label=f'Среднее: {mean_days:.0f} дней')
    plt.axvline(median_days, color='green', linestyle='dashed', linewidth=2, label=f'Медиана: {median_days:.0f} дней')
    
    plt.legend()
    plt.yscale('log') # Логарифмическая шкала для лучшей читаемости
    plt.gca().get_yaxis().set_major_formatter(plt.ScalarFormatter()) # Отключаем научную нотацию на оси Y
    
    output_path_png = report_dir / "data_coverage_histogram.png"
    plt.savefig(output_path_png)
    print(f"✅ Гистограмма сохранена в: {output_path_png}")
    # ---------------------------------------------------

    print("\n" + "="*50)
    print("--- 📊 Итоги анализа ---")
    
    print(f"\nПолный CSV отчет сохранен в файл: {output_path_csv}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализ временного покрытия данных по акциям и построение гистограммы.")
    parser.add_argument(
        '--universe_file',
        type=str,
        default='universe.csv',
        help='Файл со списком тикеров для анализа.'
    )
    args = parser.parse_args()
    main(args)