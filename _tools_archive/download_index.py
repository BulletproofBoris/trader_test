from _core.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, TOOLS_DIR, CONFIGS_DIR, ensure_dirs, load_dotenv_if_exists
from _core.libs import *
load_dotenv_if_exists()
ensure_dirs()
# _tools/download_index.py

def download_moex_index_history(years=20, data_dir="1_data"):
    """
    Скачивает дневную историю индекса IMOEX с Московской биржи,
    корректно обрабатывая пагинацию и отсутствие данных по объему.
    """
    URL = "https://iss.moex.com/iss/history/engines/stock/markets/index/boards/SNDX/securities/IMOEX.json"
    start_date = (datetime.now() - relativedelta(years=years)).strftime('%Y-%m-%d')
    print(f"Загрузка истории индекса IMOEX с {start_date}...")

    all_data = []
    start_index = 0
    
    while True:
        params = {
            'iss.meta': 'off',
            'iss.only': 'history',
            # ИЗМЕНЕНО: Удалена колонка VOLUME из запроса
            'history.columns': 'TRADEDATE,OPEN,HIGH,LOW,CLOSE',
            'limit': 100,
            'start': start_index,
            'from': start_date,
        }
        
        try:
            response = requests.get(URL, params=params)
            response.raise_for_status()
            data_chunk = response.json()['history']['data']

            if not data_chunk:
                break
            
            all_data.extend(data_chunk)
            start_index += len(data_chunk)
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Произошла ошибка при запросе: {e}"); return

    if not all_data:
        print("⚠️ MOEX ISS не вернул данные по индексу."); return

    # ИЗМЕНО: Удалена колонка 'vol' из списка
    df = pd.DataFrame(all_data, columns=['datetime', 'open', 'high', 'low', 'close'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values('datetime', inplace=True)
    
    # ИЗМЕНО: Удалена колонка 'vol' из цикла обработки
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['datetime'], keep='last', inplace=True)

    output_path = os.path.join(data_dir, "IMOEX.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Полная история индекса IMOEX ({len(df)} строк) сохранена в {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Загрузка истории индекса IMOEX.")
    parser.add_argument('--years', type=int, default=20, help='Глубина истории в годах.')
    parser.add_argument('--data_dir', type=str, default="1_data", help='Папка для сохранения данных.')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    download_moex_index_history(args.years, args.data_dir)
