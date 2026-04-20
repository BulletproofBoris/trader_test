import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from FinamPy import FinamPy
from FinamPy.grpc.accounts_service_pb2 import GetAccountRequest

# Динамически вычисляем корень проекта (на 1 уровень выше текущего файла)
# Это позволяет запускать скрипт из любой директории
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / '.env'

# Загружаем переменные, override=True нужен для перезаписи кэша терминала
load_dotenv(dotenv_path=ENV_PATH, override=True)

# Забираем токен и очищаем от возможных случайных кавычек
SECRET_TOKEN = os.getenv("FINAM_API_TOKEN")
if SECRET_TOKEN:
    SECRET_TOKEN = SECRET_TOKEN.strip('"\'')

async def main():
    if not SECRET_TOKEN:
        print("❌ Ошибка: Переменная FINAM_API_TOKEN пуста.")
        return

    print("Устанавливаем gRPC соединение с сервером Финама...")
    
    fp_provider = FinamPy(SECRET_TOKEN)
    accounts = fp_provider.account_ids
    
    if not accounts:
        print("❌ Счета не найдены. Токен отвергнут или нет прав.")
        fp_provider.close_channel()
        return
        
    print(f"✅ Успешная авторизация! Найдено счетов: {len(accounts)}\n")
    print("-" * 40)
    
    for account in accounts:
        print(f"📊 Запрашиваем данные для счета: {account}")
        
        portfolio = fp_provider.call_function(
            fp_provider.accounts_stub.GetAccount, 
            GetAccountRequest(account_id=account)
        )
        
        # Безопасно проверяем наличие данных в ответе сервера
        if portfolio and hasattr(portfolio, 'portfolio_mc') and hasattr(portfolio.portfolio_mc, 'available_cash'):
            balance = portfolio.portfolio_mc.available_cash.value
            print(f"💰 Доступные средства: {balance} RUB")
        else:
            print("💰 Баланс: 0 RUB (или данные не найдены).")
            
        print("-" * 40)

    fp_provider.close_channel()
    print("\nРабота завершена.")

if __name__ == "__main__":
    asyncio.run(main())