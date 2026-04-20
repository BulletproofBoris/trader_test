import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Подключаем корень проекта
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from _core.feature_generator import create_cross_sectional_features, create_individual_features

def load_macro_data(raw_dir: Path) -> pd.DataFrame:
    print("🌍 Сборка макроэкономических данных...")
    macro_map = {
        "MACRO_USDRUB.csv": "usdrub_close",
        "MACRO_BRENT.csv": "brent_close",
        "MACRO_SP500.csv": "sp500_close",
        "MACRO_IMOEX.csv": "imoex_close",
        "MACRO_VIX.csv": "vix_close"
    }
    
    ext_df = pd.DataFrame()
    for filename, target_col in macro_map.items():
        filepath = raw_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            val_col = df.columns[0]
            df = df.rename(columns={val_col: target_col})
            
            if ext_df.empty:
                ext_df = df
            else:
                ext_df = ext_df.join(df, how='outer')
                
    if not ext_df.empty and 'cbr_rate_close' not in ext_df.columns:
        ext_df['cbr_rate_close'] = 16.0 
        
    ext_df.index.name = 'datetime'
    return ext_df.ffill().bfill() if not ext_df.empty else ext_df

def process_single_ticker(ticker: str, df: pd.DataFrame, external_data: pd.DataFrame, cs_features: pd.DataFrame, out_dir: Path) -> tuple:
    try:
        _, df_final = create_individual_features((ticker, df), external_data, cs_features)
        df_final['ticker'] = ticker
        safe_symbol = ticker.replace('@', '_')
        df_final.to_csv(out_dir / f"{safe_symbol}_processed.csv", index=False)
        return ticker, True, "Успешно"
    except Exception as e:
        return ticker, False, str(e)

def build_dataset(max_workers: int, timeframe: str):
    RAW_DIR = BASE_DIR / "data" / "raw" / timeframe
    PROCESSED_DIR = BASE_DIR / "data" / "processed" / timeframe
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Генерация глобального датасета (Таймфрейм: {timeframe} | CPU: {max_workers})...")
    
    universe_file = BASE_DIR / "universe.csv"
    if not universe_file.exists():
        print("❌ Файл universe.csv не найден!")
        return
        
    tickers = pd.read_csv(universe_file)['Ticker'].tolist()
    all_data = {}
    
    print(f"📦 Загрузка сырых данных из {RAW_DIR.relative_to(BASE_DIR)} ({len(tickers)} бумаг)...")
    for ticker in tqdm(tickers):
        safe_symbol = ticker.replace('@', '_')
        filepath = RAW_DIR / f"{safe_symbol}_{timeframe.upper()}_MAX.csv"
        
        if filepath.exists():
            df = pd.read_csv(filepath)
            df.columns = [c.lower() for c in df.columns]
            df['datetime'] = pd.to_datetime(df.iloc[:, 0])
            df.set_index('datetime', inplace=True)
            all_data[ticker] = df

    external_data = load_macro_data(RAW_DIR)
    cs_features = create_cross_sectional_features(all_data)

    print(f"\n⚙️ Параллельный расчет технических индикаторов...")
    success_count = 0
    failed_tickers = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_ticker, ticker, df, external_data, cs_features, PROCESSED_DIR): ticker for ticker, df in all_data.items()}
        for future in tqdm(as_completed(futures), total=len(futures)):
            res_ticker, success, msg = future.result()
            if success:
                success_count += 1
            else:
                failed_tickers.append(res_ticker)
                print(f"\n🔴 Ошибка {res_ticker}: {msg}")

    print("\n" + "=" * 50)
    print("📈 ГЕНЕРАЦИЯ ДАТАСЕТОВ ЗАВЕРШЕНА")
    print(f"Успешно обработано: {success_count} / {len(all_data)}")
    print(f"📁 Файлы сохранены в: {PROCESSED_DIR.relative_to(BASE_DIR)}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--timeframe", type=str, default="1d")
    args = parser.parse_args()
    build_dataset(args.workers, args.timeframe)