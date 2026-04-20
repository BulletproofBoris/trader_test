# _tools/universal_downloader.py
from __future__ import annotations
from typing import Optional
from datetime import datetime, date
import os
import time
import json
import random
import warnings

from _core.paths import PROJECT_ROOT, ensure_dirs, load_dotenv_if_exists
from _core.libs import *
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv_if_exists()
ensure_dirs()

MOEX_BASE = "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/{board}/securities"
DEFAULT_BOARD = "TQBR"

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "trader_test/1.0"})
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def _parse_moex_json(json_text: str) -> Optional[pd.DataFrame]:
    try:
        json_data = json.loads(json_text)
        if not isinstance(json_data, list) or len(json_data) < 2: return None
        data_dict = json_data[1]
        if not isinstance(data_dict, dict): return None
        history_data = data_dict.get('history')
        if not history_data or not isinstance(history_data, list):
            return pd.DataFrame() 
        return pd.DataFrame(history_data)
    except (json.JSONDecodeError, IndexError, AttributeError):
        return None

def download_one_stock_history(
    ticker: str, years: int, data_dir: str, board: str, sleep_sec: float
) -> Tuple[str, str]:
    warnings.simplefilter('ignore', FutureWarning)
    if sleep_sec > 0:
        time.sleep(random.uniform(0, sleep_sec))

    # --- ИЗМЕНЕНИЕ 1: Новый формат имени файла ---
    output_path = Path(PROJECT_ROOT) / data_dir / f"{ticker}_D1_MOEX.csv"
    
    start_date_str = (datetime.now() - relativedelta(years=years)).strftime("%Y-%m-%d")
    existing_df = None
    
    # --- ИЗМЕНЕНИЕ 2: Логика инкрементальной загрузки ---
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            last_date_in_file = pd.to_datetime(existing_df['datetime']).max().date()
            # Запрашиваем данные, начиная со следующего дня
            start_date_str = (last_date_in_file + timedelta(days=1)).strftime('%Y-%m-%d')
        except Exception:
            # Если файл битый, скачиваем заново
            existing_df = None
            
    # Если сегодня раньше даты начала запроса, то новых данных нет
    if date.today() < datetime.strptime(start_date_str, '%Y-%m-%d').date():
        return ticker, "no_new_data"

    all_data = []
    start_cursor = 0
    session = _make_session()
    
    try:
        while True:
            url = MOEX_BASE.format(board=board) + f"/{ticker}.json"
            params = {
                'start': start_cursor, 'from': start_date_str, 'iss.json': 'extended',
                'iss.meta': 'off', 'limit': 100, 'lang': 'ru'
            }
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            df_chunk = _parse_moex_json(response.text)
            
            if df_chunk is None: return ticker, "parse_error"
            if df_chunk.empty: break

            all_data.append(df_chunk)
            start_cursor += len(df_chunk)
            if len(df_chunk) < 100: break
    except requests.RequestException:
        session.close()
        return ticker, "network_error"
    
    session.close()

    # --- Логика объединения и сохранения ---
    if not all_data:
        return ticker, "no_new_data"

    try:
        new_df = pd.concat(all_data, ignore_index=True)
        new_df.columns = [col.lower() for col in new_df.columns]
        
        rename_map = {
            'tradedate': 'datetime', 'open': 'open', 'high': 'high',
            'low': 'low', 'close': 'close', 'volume': 'volume'
        }
        required_cols = list(rename_map.keys())
        if not all(col in new_df.columns for col in required_cols):
            return ticker, "missing_cols"

        final_df = new_df[required_cols].rename(columns=rename_map)
        
        # Если были старые данные, объединяем
        if existing_df is not None:
            final_df = pd.concat([existing_df, final_df], ignore_index=True)

        # Очистка и сохранение
        final_df['datetime'] = pd.to_datetime(final_df['datetime'])
        final_df.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
        final_df.sort_values('datetime', inplace=True)
        final_df.to_csv(output_path, index=False)
        
        return ticker, "updated" if existing_df is not None else "created"
    except Exception:
        return ticker, "save_error"


def download_all_stocks(
    tickers: list[str], years: int, data_dir: str, board: str,
    processes: int, sleep_sec: float
):
    worker_func = partial(
        download_one_stock_history,
        years=years, data_dir=data_dir, board=board, sleep_sec=sleep_sec
    )
    
    results_map = {"created": 0, "updated": 0, "no_new_data": 0, "errors": 0}
    
    with multiprocessing.Pool(processes=processes) as pool:
        results = list(tqdm(pool.imap(worker_func, tickers), total=len(tickers), desc="Загрузка акций"))
        
    for ticker, status in results:
        if status in ["created", "updated", "no_new_data"]:
            if status in results_map:
                results_map[status] += 1
        else:
            results_map["errors"] += 1
            
    print("✅ Загрузка акций завершена.")
    print(f"   - Новых файлов создано: {results_map['created']}")
    print(f"   - Файлов обновлено: {results_map['updated']}")
    print(f"   - Актуальных файлов (без изменений): {results_map['no_new_data']}")
    print(f"   - Ошибок: {results_map['errors']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инкрементальная загрузка дневной истории акций MOEX.")
    parser.add_argument("--years", type=int, default=20, help="Сколько лет истории забирать для НОВЫХ файлов")
    parser.add_argument("--board", default=DEFAULT_BOARD, help="Доска (по умолчанию TQBR)")
    parser.add_argument("--data-dir", default="1_data", help="Куда сохранять CSV")
    parser.add_argument("--universe-file", default="universe.csv", help="CSV со списком тикеров")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2)//2), help="Число процессов")
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="Пауза между тикерами (внутри воркера)")
    args = parser.parse_args()

    uni_path = Path(PROJECT_ROOT) / args.universe_file
    if uni_path.exists():
        df_u = pd.read_csv(uni_path)
        col = "ticker" if "ticker" in df_u.columns else df_u.columns[0]
        tickers = df_u[col].dropna().astype(str).unique().tolist()
    else:
        print(f"[WARN] Universe-файл {uni_path} не найден.")
        tickers = []

    if tickers:
        download_all_stocks(
            tickers=tickers, years=args.years, data_dir=args.data_dir,
            board=args.board, processes=args.processes, sleep_sec=args.sleep_sec
        )