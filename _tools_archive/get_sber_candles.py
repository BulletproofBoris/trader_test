import asyncio
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
from google.protobuf.json_format import ParseDict

from FinamPy import FinamPy
from FinamPy.grpc.marketdata_service_pb2 import BarsRequest

# --- Конфигурация ---
BASE_DIR = Path("/home/restorator/trader_test")
ENV_PATH = BASE_DIR / '.env'
DATA_DIR = BASE_DIR / '1_data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(dotenv_path=ENV_PATH, override=True)
SECRET_TOKEN = os.getenv("FINAM_API_TOKEN").strip('"\'')

async def fetch_history():
    print("🚀 Запуск постраничной выгрузки SBER (по 1 году)...")
    fp = FinamPy(SECRET_TOKEN)
    symbol = "SBER@MISX" 
    
    current_time = datetime.now(timezone.utc)
    start_year = 1999 
    all_data = []

    try:
        for year in range(start_year, current_time.year + 1):
            # Начало года
            year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
            # Конец года (но не позже текущего момента)
            year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            if year_end > current_time:
                year_end = current_time

            print(f"📅 Запрос за {year} год...")

            request_data = {
                "symbol": symbol,
                "timeframe": "TIME_FRAME_D",
                "interval": {
                    "startTime": year_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "endTime": year_end.strftime("%Y-%m-%dT%H:%M:%SZ")
                }
            }
            
            request = BarsRequest()
            ParseDict(request_data, request)

            try:
                response = fp.call_function(fp.marketdata_stub.Bars, request)
                
                if response and response.bars:
                    print(f"  ✅ Получено {len(response.bars)} свечей")
                    for b in response.bars:
                        dt = datetime.fromtimestamp(b.timestamp.seconds, tz=timezone.utc)
                        all_data.append({
                            'Date': dt.date(),
                            'Open': float(b.open.value),
                            'High': float(b.high.value),
                            'Low': float(b.low.value),
                            'Close': float(b.close.value),
                            'Volume': float(b.volume.value)
                        })
                else:
                    print(f"  ⚪ Данных за {year} нет.")
            except Exception as e:
                print(f"  ❌ Ошибка за {year}: {e}")
            
            await asyncio.sleep(0.5) # Пауза между годами

        if not all_data:
            print("❌ История пуста.")
            return

        df = pd.DataFrame(all_data).drop_duplicates(subset=['Date']).sort_values('Date')
        file_path = DATA_DIR / "SBER_D1_MAX.csv"
        df.to_csv(file_path, index=False)
        
        print("-" * 40)
        print(f"✨ ГОТОВО! Всего строк: {len(df)}")
        print(f"📈 Диапазон: {df['Date'].min()} — {df['Date'].max()}")
        print("-" * 40)

    finally:
        fp.close_channel()

if __name__ == "__main__":
    asyncio.run(fetch_history())