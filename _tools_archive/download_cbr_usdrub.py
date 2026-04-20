# _tools/download_cbr_usdrub.py
from _core.paths import DATA_DIR, ensure_dirs
from _core.libs import *
import argparse
import xml.etree.ElementTree as ET

ensure_dirs()

def download_cbr_usdrub_history(years=20, data_dir=DATA_DIR):
    """
    Загружает официальный курс USD/RUB с сайта ЦБ РФ за указанный период.
    """
    end_date = datetime.now()
    start_date = end_date - relativedelta(years=years)
    
    start_date_str = start_date.strftime('%d/%m/%Y')
    end_date_str = end_date.strftime('%d/%m/%Y')
    
    # ID курса доллара США в справочнике ЦБ
    usd_id = 'R01235'
    
    url = f"http://www.cbr.ru/scripts/XML_dynamic.asp?date_req1={start_date_str}&date_req2={end_date_str}&VAL_NM_RQ={usd_id}"
    
    print(f"Загрузка истории USD/RUB с сайта ЦБ РФ ({start_date_str} - {end_date_str})...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        
        all_data = []
        for record in root.findall('Record'):
            date_str = record.get('Date')
            value_str = record.find('Value').text
            
            date = datetime.strptime(date_str, '%d.%m.%Y')
            value = float(value_str.replace(',', '.'))
            all_data.append({'datetime': date, 'close': value})

        if not all_data:
            print("⚠️ ЦБ РФ не вернул данные по курсу USD/RUB.")
            return

        df = pd.DataFrame(all_data)
        
        # Создаем полный диапазон дат для непрерывного ряда
        full_date_range = pd.date_range(start=df['datetime'].min(), end=datetime.now(), freq='D')
        full_df = pd.DataFrame(full_date_range, columns=['datetime'])
        
        # Объединяем и заполняем пропуски на выходных (forward fill)
        merged_df = pd.merge(full_df, df, on='datetime', how='left').ffill()

        # Добавляем "пустые" OHLCV колонки для совместимости с форматом
        for col in ['open', 'high', 'low', 'volume']:
            if col not in merged_df.columns:
                if col == 'volume':
                    merged_df[col] = 0.0
                else:
                    merged_df[col] = merged_df['close']

        output_path = data_dir / "USDRUB.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"✅ Полная история USD/RUB ({len(merged_df)} строк) сохранена в {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Сетевая ошибка при запросе к API ЦБ РФ: {e}")
    except ET.ParseError:
        print("❌ Ошибка парсинга XML ответа от ЦБ РФ. Возможно, изменился формат.")
    except Exception as e:
        print(f"❌ Произошла непредвиденная ошибка: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Загрузка истории официального курса USD/RUB от ЦБ РФ.")
    parser.add_argument('--years', type=int, default=20, help='Глубина истории в годах.')
    args = parser.parse_args()
    download_cbr_usdrub_history(years=args.years)