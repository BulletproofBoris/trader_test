# _tools/plot_ticker.py
import argparse
from pathlib import Path
import warnings
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style

# --- Настройка окружения и путей ---
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
style.use('dark_background')

# --- Импорты из вашего проекта ---
try:
    from _core.paths import DATA_DIR, RESULTS_DIR, ensure_dirs
    from _core.data_loader import load_data
except ImportError:
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from _core.paths import DATA_DIR, RESULTS_DIR, ensure_dirs
    from _core.data_loader import load_data

def main(args):
    print(f"--- 📈 Построение графика для тикера '{args.ticker}' ---")

    # --- 1. Загрузка данных из папки 1_data ---
    df = load_data(args.ticker)

    if df is None or df.empty:
        print(f"❌ Ошибка: Не удалось загрузить данные для тикера '{args.ticker}'.")
        print(f"   Проверьте, что файл с данными существует в папке: {DATA_DIR}")
        return

    # --- 2. Фильтрация данных за последний год ---
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    end_date = df['datetime'].max()
    start_date = end_date - pd.DateOffset(years=args.years)
    df_plot = df[df['datetime'] >= start_date].copy()

    if df_plot.empty:
        print(f"❌ Ошибка: Не найдено данных за последний год для тикера '{args.ticker}'.")
        return
        
    print(f"Найдено {len(df_plot)} записей за период с {start_date.date()} по {end_date.date()}.")

    # --- 3. Визуализация ---
    print("Создание графика...")
    fig, ax = plt.subplots(figsize=(20, 10))
    
    ax.plot(df_plot['datetime'], df_plot['close'], label=f'Цена Close для {args.ticker.upper()}', color='deepskyblue', linewidth=1.5)
    
    ax.set_title(f"Цена акции {args.ticker.upper()} за последний год (источник: 1_data)", fontsize=18, pad=20)
    ax.set_ylabel("Цена", fontsize=14)
    ax.set_xlabel("Дата", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    
    # --- 4. Сохранение результата ---
    report_dir = RESULTS_DIR / "reports"
    report_dir.mkdir(exist_ok=True, parents=True)
    output_path = report_dir / f"price_chart_{args.ticker}.png"
    plt.savefig(output_path)
    
    print(f"✅ График успешно сохранен в: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Построение графика цены для выбранного тикера из папки 1_data.")
    parser.add_argument("--ticker", type=str, required=True, help="Тикер акции для построения графика (напр. SBER).")
    parser.add_argument("--years", type=int, default=1, help="Количество лет для отображения на графике.")
    
    args = parser.parse_args()
    main(args)