# _tools/update_universe.py
import argparse
import pandas as pd
from functools import partial
import time
import json
import os
import requests

# Импортируем все необходимое из вашего проекта
from _core.paths import ensure_dirs, RESULTS_DIR
from _core.libs import tqdm, multiprocessing

ensure_dirs()

# --- Константы, чтобы скрипт был самодостаточным ---
MOEX_BASE = "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/{board}/securities"
DEFAULT_BOARD = "TQBR"

def _make_session() -> requests.Session:
    """Создает сессию для HTTP-запросов."""
    s = requests.Session()
    s.headers.update({"User-Agent": "trader_test/1.0"})
    return s

def get_initial_moex_tickers():
    """
    Получает полный список торгуемых акций с Мосбиржи.
    """
    print("Запрашиваю актуальный список акций с Московской биржи...")
    base_url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
    params = {"iss.meta": "off", "iss.only": "securities,securities.cursor", 'iss.json': 'extended'}
    all_tickers = []
    start = 0
    
    with _make_session() as session:
        while True:
            try:
                current_params = params.copy()
                current_params['securities.cursor'] = start
                
                response = session.get(base_url, params=current_params, timeout=20)
                response.raise_for_status()
                
                data = response.json()

                if not isinstance(data, list) or len(data) < 2:
                    break
                
                data_block = data[1]
                
                securities_list = data_block.get('securities')
                if securities_list and isinstance(securities_list, list):
                    for row in securities_list:
                        if isinstance(row, dict) and 'SECID' in row:
                            all_tickers.append(str(row['SECID']))
                else:
                    break
                
                cursor_list = data_block.get('securities.cursor')
                if cursor_list and isinstance(cursor_list, list) and cursor_list:
                    cursor_data = cursor_list[0]
                    if isinstance(cursor_data, dict):
                        total = cursor_data.get('TOTAL', 0)
                        page_size = cursor_data.get('PAGESIZE', 100)
                        start += page_size
                        
                        if start >= total:
                            break
                    else:
                        break
                else:
                    break
                
                time.sleep(0.25)
                
            except requests.RequestException as e:
                print(f"⚠️  Сетевая ошибка при получении списка акций: {e}. Прерываю.")
                return []
            except Exception as e:
                print(f"⚠️  Критическая ошибка при получении списка: {e}. Прерываю.")
                return []

    unique_tickers = sorted(list(set(all_tickers)))
    print(f"✅ Получено {len(unique_tickers)} уникальных тикеров.")
    return unique_tickers


def get_history_days_count(ticker: str, session: requests.Session) -> dict:
    """
    Для одного тикера запрашивает метаданные истории и возвращает количество дней.
    """
    url = MOEX_BASE.format(board=DEFAULT_BOARD) + f"/{ticker}.json"
    params = {'limit': 1, 'iss.json': 'extended', 'iss.meta': 'off'}
    
    try:
        response = session.get(url, params=params, timeout=15)
        if response.status_code != 200:
            return {"ticker": ticker, "history_days": 0, "error": f"HTTP status {response.status_code}"}
            
        data = response.json()
        
        if isinstance(data, list) and len(data) > 1:
            history_cursor_block = data[1].get('history.cursor')
            if history_cursor_block and isinstance(history_cursor_block, list) and history_cursor_block:
                try:
                    total_days = history_cursor_block[0].get('TOTAL', 0)
                    return {"ticker": ticker, "history_days": int(total_days), "error": None}
                except (ValueError, IndexError):
                    return {"ticker": ticker, "history_days": 0, "error": "Could not parse 'TOTAL' from data"}
        
        return {"ticker": ticker, "history_days": 0, "error": "Unexpected JSON structure"}
    
    except requests.exceptions.RequestException as e:
        return {"ticker": ticker, "history_days": 0, "error": f"Network error: {type(e).__name__}"}
    except json.JSONDecodeError:
        return {"ticker": ticker, "history_days": 0, "error": "Invalid JSON response"}
    except Exception as e:
        return {"ticker": ticker, "history_days": 0, "error": f"Unexpected error: {e}"}


def main(args):
    """Основной пайплайн: получение списка, проверка истории, фильтрация, сохранение."""
    
    initial_tickers = get_initial_moex_tickers()
    if not initial_tickers:
        print("❌ Не удалось получить список акций. Завершаю работу.")
        return

    print(f"\n--- 🕵️ Проверка длины истории для {len(initial_tickers)} тикеров ---")
    
    with _make_session() as session:
        worker_func = partial(get_history_days_count, session=session)
        
        with multiprocessing.Pool(processes=args.workers) as pool:
            results = list(tqdm(pool.imap(worker_func, initial_tickers), total=len(initial_tickers), desc="Анализ истории"))

    report_df = pd.DataFrame(results)
    
    # --- ИСПРАВЛЕНИЕ: Сначала очищаем данные, потом сортируем и фильтруем ---
    # 1. Принудительно конвертируем в числа, все ошибки (нечисловые значения) станут NaN
    report_df['history_days'] = pd.to_numeric(report_df['history_days'], errors='coerce')
    # 2. Заполняем все NaN (включая те, что появились после to_numeric) нулями и приводим к целому числу
    report_df['history_days'] = report_df['history_days'].fillna(0).astype(int)
    
    # 3. Теперь, когда колонка чистая, сортируем
    report_df.sort_values('history_days', ascending=False, inplace=True)
    
    report_path = RESULTS_DIR / "universe_full_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"\n📊 Полный отчет о длине истории сохранен в: {report_path}")
    
    errors_df = report_df[report_df['error'].notna()]
    if not errors_df.empty:
        print("\n--- ⚠️ Обнаружены проблемы при проверке тикеров ---")
        error_counts = errors_df['error'].value_counts()
        print(error_counts.to_string())
        print("----------------------------------------------------")

    # 4. Теперь фильтрация будет работать корректно
    filtered_df = report_df[report_df['history_days'] >= args.min_days]
    filtered_tickers = filtered_df['ticker'].tolist()

    print("\n" + "="*50)
    print("--- 🔬 Итоги фильтрации ---")
    print(f"Всего проанализировано: {len(report_df)} тикеров")
    print(f"Прошли фильтр (история >= {args.min_days} дней): {len(filtered_tickers)} тикеров")
    print(f"Отброшено: {len(report_df) - len(filtered_tickers)} тикеров")
    print("="*50)
    
    if filtered_tickers:
        universe_df = pd.DataFrame(filtered_tickers, columns=['ticker'])
        universe_df.to_csv(args.output_file, index=False)
        print(f"\n✅ Обновленный и отфильтрованный `universe.csv` сохранен.")
    else:
        print(f"\n⚠️ После фильтрации не осталось ни одного тикера. Файл `{args.output_file}` не был сохранен.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обновление и фильтрация списка акций MOEX по длине истории.")
    parser.add_argument('--min-days', type=int, default=252, help="Минимальное количество торговых дней в истории.")
    parser.add_argument('--output-file', type=str, default='universe.csv', help="Имя выходного файла со списком тикеров.")
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help="Количество параллельных процессов.")
    
    args = parser.parse_args()
    main(args)