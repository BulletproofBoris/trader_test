# _tools/apply_splits.py
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import yfinance as yf
import time
import warnings
import os
from concurrent.futures import ThreadPoolExecutor

# --- Подавляем FutureWarning от pandas ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# --- Секция для ручных правил ---
# ==============================================================================
def patch_VEON_RX(df: pd.DataFrame) -> pd.DataFrame:
    # ... (код этой функции без изменений) ...
    print("    -> Применяю ручное правило для VEON-RX...")
    price_cols = ['open', 'high', 'low', 'close']
    volume_col = 'volume'
    split_date = pd.to_datetime("2023-03-08")
    ratio = 25.0
    df_before = df[df.index < split_date]
    if not df_before.empty and df_before['close'].median() > 100:
         print("    (i) Данные VEON-RX, вероятно, уже исправлены. Пропускаю.")
         return df
    print(f"    (i) Применяю корректировку: Консолидация с множителем {ratio} от {split_date.date()}")
    mask = df.index < split_date
    df.loc[mask, price_cols] *= ratio
    df.loc[mask, volume_col] /= ratio
    print("    ✅ Корректировка VEON-RX завершена.")
    return df

MANUAL_PATCH_MAP = {
    'VEON-RX': patch_VEON_RX,
}

# ==============================================================================
# --- Секция автоматического поиска через yfinance (теперь многопоточная) ---
# ==============================================================================

def _fetch_for_one_ticker(ticker: str) -> list:
    """Вспомогательная функция для загрузки данных по одному тикеру."""
    try:
        yahoo_ticker_str = f"{ticker}.ME"
        yf_ticker = yf.Ticker(yahoo_ticker_str)
        splits = yf_ticker.splits
        if splits.empty:
            return []
        
        ticker_splits = []
        for date, yf_ratio in splits.items():
            price_multiplier = 1 / yf_ratio
            event_type = 'Консолидация' if price_multiplier > 1 else 'Дробление'
            ticker_splits.append({
                'ticker': ticker,
                'split_date': date.tz_convert(None).normalize(),
                'ratio': price_multiplier,
                'type': event_type
            })
        return ticker_splits
    except Exception:
        return []

def fetch_splits_from_yfinance(tickers: list[str], max_workers: int) -> pd.DataFrame:
    """
    Загружает историю сплитов с Yahoo Finance в многопоточном режиме.
    """
    print(f" -> Загрузка истории корпоративных событий с Yahoo Finance ({max_workers} потоков)...")
    all_splits_data = []
    
    yf.set_tz_cache_location(os.devnull)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(_fetch_for_one_ticker, tickers), total=len(tickers), desc="Проверка акций на Yahoo Finance"))

    # Собираем результаты из всех потоков
    for result_list in results:
        all_splits_data.extend(result_list)
            
    if not all_splits_data:
        print("    (w) yfinance не нашел ни одного события сплита/консолидации.")
        return pd.DataFrame()
        
    final_df = pd.DataFrame(all_splits_data)
    final_df['split_date'] = pd.to_datetime(final_df['split_date'])
    final_df = final_df.sort_values(by='split_date').reset_index(drop=True)
    
    print(f"\n    ✅ yfinance нашел {len(final_df)} событий. История загружена.")
    print(final_df)
    return final_df

def adjust_df_auto(df: pd.DataFrame, ticker_splits: pd.DataFrame) -> pd.DataFrame:
    # ... (код этой функции без изменений) ...
    price_cols = ['open', 'high', 'low', 'close']
    volume_col = 'volume'
    ticker_splits = ticker_splits.sort_values(by='split_date')
    for _, split_info in ticker_splits.iterrows():
        split_date = split_info['split_date']
        ratio = split_info['ratio']
        df_before = df[df.index < split_date].tail(30)
        df_after = df[df.index >= split_date].head(30)
        if not df_before.empty and not df_after.empty:
            median_before = df_before['close'].median()
            median_after = df_after['close'].median()
            if median_before > 0 and abs(median_before - median_after) / median_after < 0.5:
                 print(f"    (i) Данные для события от {split_date.date()}, вероятно, уже скорректированы. Пропускаю.")
                 continue
        print(f"    (i) Применяю авто-корректировку: {split_info['type']} с множителем {ratio:.4f} от {split_date.date()}")
        mask_to_patch = df.index.normalize() < split_date.normalize()
        df.loc[mask_to_patch, price_cols] *= ratio
        df.loc[mask_to_patch, volume_col] /= ratio
    return df

# ==============================================================================
# --- Основная логика ---
# ==============================================================================
def main(args):
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"❌ Ошибка: Папка {data_dir} не найдена.")
        return
    try:
        universe_df = pd.read_csv(args.universe_file)
        tickers = universe_df['ticker'].tolist()
    except Exception as e:
        print(f"❌ Не удалось прочитать файл {args.universe_file}: {e}")
        return
        
    # <-- Передаем количество потоков в функцию
    auto_splits_df = fetch_splits_from_yfinance(tickers, args.workers)
    
    if auto_splits_df.empty and not MANUAL_PATCH_MAP:
        print("⚠️ Не найдено информации о сплитах, дальнейшая обработка пропускается.")
        return
        
    files_to_check = list(data_dir.glob("*.csv"))
    for file_path in tqdm(files_to_check, desc="Применение корректировок"):
        ticker = file_path.stem.split('_')[0]
        df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')
        df = df.sort_index()
        if df.empty:
            continue
        df_original = df.copy()
        is_manual = False
        if ticker in MANUAL_PATCH_MAP:
            print(f"\nНайден тикер для РУЧНОГО исправления: {ticker}")
            df = MANUAL_PATCH_MAP[ticker](df)
            is_manual = True
        ticker_auto_splits = auto_splits_df[auto_splits_df['ticker'] == ticker]
        if not ticker_auto_splits.empty:
            if not is_manual:
                print(f"\nНайдены АВТО-корректировки для {ticker}:")
            df = adjust_df_auto(df, ticker_auto_splits)
        if not df_original.equals(df):
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = df[price_cols].apply(pd.to_numeric, errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
            df = df.ffill().bfill()
            df['volume'] = df['volume'].round().astype(np.int64)
            df.to_csv(file_path)
            print(f"    ✅ Файл {file_path.name} скорректирован и сохранен.")

    print("\n--- ✅ Процесс гибридной корректировки данных завершен ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Автоматическая и ручная корректировка данных на сплиты.")
    parser.add_argument("--data-dir", type=str, default="1_data")
    parser.add_argument("--universe-file", type=str, default="universe.csv")
    # <-- Новый аргумент
    parser.add_argument("--workers", type=int, default=10, help="Количество потоков для загрузки данных с yfinance.")
    args = parser.parse_args()
    main(args)