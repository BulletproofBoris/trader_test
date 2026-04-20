# _tools/analyze_volatility_for_labeling.py
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

# --- Прямое определение путей ---
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "1_data"
UNIVERSE_FILE = PROJECT_DIR / "universe.csv"
# ---

def run_data_quality_report(horizon: int, sma_window: int, deviation_threshold: float):
    """
    Проводит полный анализ качества данных:
    1. Ищет выбросы как аномальные отклонения от скользящего среднего.
    2. Рассчитывает статистику для TP/SL.
    """
    try:
        tickers = pd.read_csv(UNIVERSE_FILE, header=0)['ticker'].tolist()
    except (FileNotFoundError, KeyError):
        print(f"Ошибка: Не удалось прочитать тикеры из файла {UNIVERSE_FILE}")
        return

    all_tp_returns = []
    all_sl_returns = []
    outlier_details = []

    print(f"Анализ качества данных для {len(tickers)} акций...")
    
    for ticker in tqdm(tickers, desc="Анализ акций"):
        # Умный поиск файла
        found_path = None
        possible_suffixes = ['.csv', '_D1_MOEX.csv', '_D1.csv']
        for suffix in possible_suffixes:
            path_candidate = DATA_DIR / f"{ticker}{suffix}"
            if path_candidate.exists():
                found_path = path_candidate
                break
        if not found_path:
            continue

        df = pd.read_csv(found_path, parse_dates=['datetime'])
        if df.empty or len(df) <= horizon or len(df) <= sma_window:
            continue
            
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # --- НОВЫЙ МЕТОД ПОИСКА ВЫБРОСОВ ---
        if deviation_threshold > 0:
            df['sma'] = df['close'].rolling(window=sma_window, min_periods=1).mean()
            # Сравниваем high текущего дня с SMA предыдущего дня
            df['prev_sma'] = df['sma'].shift(1)
            df['deviation_ratio'] = df['high'] / df['prev_sma']
            
            outlier_df = df[df['deviation_ratio'] > deviation_threshold]
            
            for _, row in outlier_df.iterrows():
                outlier_details.append({
                    'ticker': ticker,
                    'date': row['datetime'].date(),
                    'reason': f"High ({row['high']:.2f}) > {deviation_threshold:.1f}x SMA ({row['prev_sma']:.2f})"
                })
        # --- КОНЕЦ НОВОГО МЕТОДА ---

        # --- Расчет статистики для TP/SL (старая логика) ---
        highs = df['high'].to_numpy()
        lows = df['low'].to_numpy()
        closes = df['close'].to_numpy()
        
        for i in range(len(df) - horizon):
            entry_price = closes[i]
            if pd.isna(entry_price) or entry_price == 0:
                continue
            
            future_window_highs = highs[i + 1 : i + 1 + horizon]
            future_window_lows = lows[i + 1 : i + 1 + horizon]
            
            potential_tp = (np.max(future_window_highs) - entry_price) / entry_price
            potential_sl = (np.min(future_window_lows) - entry_price) / entry_price
            
            all_tp_returns.append(potential_tp)
            all_sl_returns.append(potential_sl)

    # --- Вывод отчета по найденным выбросам ---
    if outlier_details:
        print("\n\n" + "="*50)
        print("🚀 Найдены аномальные отклонения от SMA (выбросы) 🚀")
        print("="*50)
        outliers_df = pd.DataFrame(outlier_details)
        print(outliers_df.sort_values(by=['ticker', 'date']).to_string())
    else:
        print("\n\n" + "="*50)
        print("✅ Аномальных отклонений от SMA не найдено.")
        print("="*50)

    # --- Расчет и вывод основной статистики ---
    if not all_tp_returns:
        print("\nНе удалось собрать данные для анализа.")
        return

    positive_moves = np.array([r for r in all_tp_returns if r > 0 and np.isfinite(r)])
    negative_moves = np.array([r for r in all_sl_returns if r < 0 and np.isfinite(r)])
    
    # ... (остальной код вывода статистики без изменений) ...
    print("\n" + "="*50)
    print("📈 Статистика по потенциальным ВЕРХНИМ движениям (для Take Profit)")
    print("="*50)
    if len(positive_moves) > 0:
        print(f"Матожидание (среднее): {positive_moves.mean():.4f} ({positive_moves.mean():.2%})")
        print(f"Стандартное отклонение: {positive_moves.std():.4f} ({positive_moves.std():.2%})")
        print(f"Медиана (50-й перцентиль): {np.median(positive_moves):.4f} ({np.median(positive_moves):.2%})")
        print(f"75-й перцентиль: {np.percentile(positive_moves, 75):.4f} ({np.percentile(positive_moves, 75):.2%})")
        print(f"95-й перцентиль: {np.percentile(positive_moves, 95):.4f} ({np.percentile(positive_moves, 95):.2%})")
    else:
        print("Положительные движения не найдены.")
    print("\n" + "="*50)
    print("📉 Статистика по потенциальным НИЖНИМ движениям (для Stop Loss)")
    print("="*50)
    if len(negative_moves) > 0:
        negative_moves_abs = np.abs(negative_moves)
        print(f"Матожидание (среднее): {negative_moves_abs.mean():.4f} ({negative_moves_abs.mean():.2%})")
        print(f"Стандартное отклонение: {negative_moves_abs.std():.4f} ({negative_moves_abs.std():.2%})")
        print(f"Медиана (50-й перцентиль): {np.median(negative_moves_abs):.4f} ({np.median(negative_moves_abs):.2%})")
        print(f"75-й перцентиль: {np.percentile(negative_moves_abs, 75):.4f} ({np.percentile(negative_moves_abs, 75):.2%})")
    else:
        print("Отрицательные движения не найдены.")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализ качества данных: волатильность и поиск выбросов.")
    parser.add_argument('--horizon', type=int, default=10, help='Горизонт в днях для анализа волатильности.')
    # --- НОВЫЕ АРГУМЕНТЫ ---
    parser.add_argument('--sma-window', type=int, default=10, help='Окно для скользящего среднего при поиске выбросов.')
    parser.add_argument('--deviation-threshold', type=float, default=2.0, help='Порог отклонения от SMA (2.0 = в 2 раза выше).')
    args = parser.parse_args()
    
    run_data_quality_report(args.horizon, args.sma_window, args.deviation_threshold)