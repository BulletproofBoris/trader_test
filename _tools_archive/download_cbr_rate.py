# _tools/download_cbr_rate.py
from _core.paths import DATA_DIR, ensure_dirs
from _core.libs import *
import argparse

ensure_dirs()

def download_and_save_cbr_key_rate(data_dir=DATA_DIR):
    """
    Скачивает ПОЛНУЮ историю ключевой ставки с сайта ЦБ РФ.
    """
    URL = "https://www.cbr.ru/hd_base/keyrate/"
    print(f"Загрузка полной истории ключевой ставки со страницы: {URL}...")

    try:
        start_date_str = "01.01.1992"
        end_date_str = datetime.now().strftime('%d.%m.%Y')
        payload = {
            "UniDbQuery.Posted": "True", "UniDbQuery.From": start_date_str, "UniDbQuery.To": end_date_str
        }
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.post(URL, data=payload, headers=headers, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(response.content)
        if not tables:
            print("⚠️ Не найдено таблиц на странице ЦБ РФ."); return

        df = tables[0]
        df.columns = ['datetime', 'key_rate']
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y')
        df['key_rate'] = pd.to_numeric(df['key_rate'], errors='coerce')

        full_date_range = pd.date_range(start=df['datetime'].min(), end=datetime.now(), freq='D')
        full_df = pd.DataFrame(full_date_range, columns=['datetime'])
        merged_df = pd.merge(full_df, df, on='datetime', how='left')
        
        # --- ИСПРАВЛЕНО: Используем более надежный метод ffill ---
        merged_df['key_rate'] = merged_df['key_rate'].ffill()
        
        output_path = data_dir / "CBR_KEY_RATE.csv"
        merged_df[['datetime', 'key_rate']].to_csv(output_path, index=False)
        print(f"✅ Полная история ключевой ставки успешно сохранена в {output_path}")

    except Exception as e:
        print(f"❌ Не удалось загрузить данные по ключевой ставке. Ошибка: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Загрузка ПОЛНОЙ истории ключевой ставки ЦБ РФ.")
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR), help="Папка для сохранения данных.")
    args = parser.parse_args()
    download_and_save_cbr_key_rate(data_dir=Path(args.data_dir))