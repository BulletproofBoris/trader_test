# _tools/download_yfinance.py
from _core.paths import DATA_DIR, ensure_dirs
from _core.libs import *
import argparse
import yfinance as yf

ensure_dirs()

def download_yfinance_history(ticker: str, output_filename: str, years: int = 20, data_dir: Path = DATA_DIR):
    """
    Скачивает историю для любого тикера с Yahoo Finance, надежно обрабатывая данные.
    """
    start_date = (datetime.now() - relativedelta(years=years)).strftime('%Y-%m-%d')
    print(f"Загрузка истории для '{ticker}' с {start_date}...")
    
    try:
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
        
        if df.empty:
            print(f"⚠️ Yahoo Finance не вернул данные для тикера '{ticker}'.")
            return

        # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Новая, более надежная логика обработки ---
        # 1. Сбрасываем индекс, чтобы дата стала колонкой
        df = df.reset_index()

        # 2. "Выравниваем" многоуровневые колонки, если они есть
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

        # 3. Приводим ВСЕ имена колонок к нижнему регистру
        df.columns = [str(col).lower() for col in df.columns]

        # 4. Переименовываем 'date' (или 'date_') в наш стандарт 'datetime'
        if 'date_' in df.columns:
            df.rename(columns={"date_": "datetime"}, inplace=True)
        elif 'date' in df.columns:
            df.rename(columns={"date": "datetime"}, inplace=True)

        # 5. Ищем и переименовываем колонки цен, удаляя суффиксы тикера
        clean_ticker_suffix = f"_{ticker.lower()}"
        rename_map = {}
        for col in df.columns:
            if col.endswith(clean_ticker_suffix):
                rename_map[col] = col.replace(clean_ticker_suffix, "")
        df.rename(columns=rename_map, inplace=True)

        # 6. Проверяем наличие 'close', предпочитая 'adj close'
        if 'adj close' in df.columns:
            df.rename(columns={'adj close': 'close'}, inplace=True)
        
        if 'close' not in df.columns:
            print(f"❌ В данных от yfinance для тикера '{ticker}' отсутствует колонка 'Close' или 'Adj Close'. Загрузка невозможна.")
            print(f"   Доступные колонки после очистки: {df.columns.tolist()}")
            return

        # 7. Убеждаемся, что есть все нужные колонки
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                if col == 'volume': df[col] = 0.0
                elif col in ['open', 'high', 'low']: df[col] = df['close']
        
        df = df[required_cols]
        df['datetime'] = pd.to_datetime(df['datetime'])

        # 8. Используем resample для создания непрерывного ряда
        df = df.set_index('datetime').resample('D').ffill().reset_index()

        output_path = data_dir / output_filename
        df.to_csv(output_path, index=False)
        print(f"✅ История '{ticker}' ({len(df)} строк) сохранена в {output_path}")

    except Exception as e:
        print(f"❌ Произошла ошибка при загрузке '{ticker}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Загрузка истории с Yahoo Finance.")
    parser.add_argument('--ticker', type=str, required=True, help='Тикер (напр. BZ=F).')
    parser.add_argument('--filename', type=str, required=True, help='Имя выходного CSV файла (напр. BRENT.csv).')
    parser.add_argument('--years', type=int, default=20, help='Глубина истории в годах.')
    args = parser.parse_args()
    
    download_yfinance_history(ticker=args.ticker, output_filename=args.filename, years=args.years)