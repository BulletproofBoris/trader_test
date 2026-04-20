# _tools/download_all_data.py
from _core.paths import PROJECT_ROOT, ensure_dirs
from _core.libs import *
import argparse

# Импортируем все необходимые функции-загрузчики
from _tools_archive.universal_downloader import download_all_stocks
from _tools_archive.download_index import download_moex_index_history
from _tools_archive.download_cbr_rate import download_and_save_cbr_key_rate
from _tools_archive.download_cbr_usdrub import download_cbr_usdrub_history
from _tools_archive.download_yfinance import download_yfinance_history

ensure_dirs()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Полная загрузка всех необходимых данных для проекта.")
    parser.add_argument("--years", type=int, default=20, help="Сколько лет истории забирать")
    parser.add_argument("--board", default="TQBR", help="MOEX board, по умолчанию TQBR")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2)//2), help="Параллелизм для акций")
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="Пауза между тикерами (сек) внутри воркера")
    parser.add_argument("--data-dir", default="1_data", help="Куда сохранять CSV")
    parser.add_argument("--universe-file", default="universe.csv", help="CSV со списком тикеров (столбец 'ticker')")
    args = parser.parse_args()

    # Загружаем universe
    uni_path = Path(PROJECT_ROOT) / args.universe_file
    if uni_path.exists():
        df = pd.read_csv(uni_path)
        col = "ticker" if "ticker" in df.columns else df.columns[0]
        tickers = df[col].dropna().astype(str).unique().tolist()
    else:
        print(f"[WARN] Файл {uni_path} не найден. Список тикеров пуст.")
        tickers = []

    print("\n--- Начинаю полную загрузку данных за {} лет ---".format(args.years))

    print("\n[1/7] Загрузка котировок по акциям...")
    download_all_stocks(
        tickers=tickers,
        years=args.years,
        data_dir=args.data_dir,
        board=args.board,
        processes=args.processes,
        sleep_sec=args.sleep_sec,
    )

    print("\n[2/7] Загрузка индексов МосБиржи (IMOEX/MOEXBC)...")
    download_moex_index_history(years=args.years)

    print("\n[3/7] Загрузка ключевой ставки ЦБ РФ...")
    download_and_save_cbr_key_rate()
    
    print("\n[4/7] Загрузка истории официального курса USD/RUB от ЦБ РФ...")
    download_cbr_usdrub_history(years=args.years)
    
    print("\n[5/7] Загрузка истории нефти Brent (BZ=F)...")
    download_yfinance_history(ticker="BZ=F", output_filename="BRENT.csv", years=args.years)
    
    print("\n[6/7] Загрузка истории индекса S&P 500 (^GSPC)...")
    download_yfinance_history(ticker="^GSPC", output_filename="SP500.csv", years=args.years)

    print("\n[7/7] Загрузка истории индекса VIX (^VIX)...")
    download_yfinance_history(ticker="^VIX", output_filename="VIX.csv", years=args.years)
    
    print("\n--- ✅ Все данные успешно загружены и обновлены ---")
