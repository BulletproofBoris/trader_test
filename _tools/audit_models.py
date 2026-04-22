import os
import pandas as pd
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent
RL_DIR = BASE_DIR / "data" / "processed" / "2000_2026_1d" / "rl_env"
RESULTS_DIR = RL_DIR / "ray_results" / "pbt_trading_bot"
SUMMARY_FILE = RL_DIR / "training_summary.txt"
LOG_FILE = RL_DIR / "audit_log.txt"

def classify_model(train_ret, test_ret):
    if test_ret > 150 or (train_ret > 0 and test_ret > train_ret * 5):
        return "🚨 Подозрительный (Reward Hack)"
    elif train_ret > 5 and test_ret < 0:
        return "⚠️ Переобучен (Overfit)"
    elif train_ret > 2 and test_ret > 0:
        return "✅ ПЕРСПЕКТИВНЫЙ (Robust)"
    else:
        return "💤 Слабый (Underfit)"

def get_latest_checkpoint(trial_id):
    """Ищет самую свежую папку checkpoint_... для заданного trial_id"""
    for item in os.listdir(RESULTS_DIR):
        if trial_id in item and os.path.isdir(os.path.join(RESULTS_DIR, item)):
            trial_path = os.path.join(RESULTS_DIR, item)
            checkpoints = [d for d in os.listdir(trial_path) if d.startswith("checkpoint_")]
            if checkpoints:
                # Сортируем по номеру итерации в названии
                checkpoints.sort(key=lambda x: int(re.search(r'\d+', x).group()))
                latest_cp = checkpoints[-1]
                iteration = int(re.search(r'\d+', latest_cp).group())
                return trial_path, iteration
    return None, None

def audit_models():
    data = []
    print("🔍 Чтение статистики из training_summary.txt и поиск чекпоинтов...")
    
    if not SUMMARY_FILE.exists():
        print(f"❌ Файл {SUMMARY_FILE} не найден. Сначала запустите обучение.")
        return pd.DataFrame()

    with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Парсим нашу текстовую таблицу
    start_parsing = False
    for line in lines:
        if line.startswith("---"):
            start_parsing = True
            continue
        if start_parsing:
            if not line.strip() or line.startswith("*"):
                continue # Конец таблицы
            
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                trial_id = parts[0]
                status = parts[1]
                
                try:
                    train_ret = float(parts[2])
                    test_ret = float(parts[3])
                except ValueError:
                    continue # Пропускаем "WAITING..." и прочее
                
                # Ищем физическую папку чекпоинта для этого trial_id
                trial_path, iteration = get_latest_checkpoint(trial_id)
                
                if trial_path and iteration:
                    data.append({
                        "Trial_ID": trial_id,
                        "Iteration": iteration,
                        "Train_Ret": round(train_ret, 2),
                        "Test_Ret": round(test_ret, 2),
                        "Path": trial_path
                    })

    if not data:
        print("❌ Не удалось извлечь данные или найти физические чекпоинты на диске.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["Status"] = df.apply(lambda row: classify_model(row["Train_Ret"], row["Test_Ret"]), axis=1)
    
    # Берем топ-15 из тех, что реально существуют
    top_models = df[df["Status"].str.contains("ПЕРСПЕКТИВНЫЙ")].sort_values("Test_Ret", ascending=False).head(15)

    if top_models.empty:
        print("⚠️ 'Идеальных' моделей не найдено. Берем все доступные...")
        top_models = df.sort_values("Test_Ret", ascending=False).head(15)

    output = []
    output.append("="*80 + f"\n{'ДОСТУПНЫЕ ЧЕКПОИНТЫ ДЛЯ ТУРНИРА':^80}\n" + "="*80)
    format_str = "{:<15} | {:<5} | {:<10} | {:<10} | {:<25}"
    output.append(format_str.format("Trial ID", "Iter", "Train %", "Test %", "Status"))
    output.append("-" * 80)
    for _, row in top_models.iterrows():
        output.append(format_str.format(row["Trial_ID"], row["Iteration"], row["Train_Ret"], row["Test_Ret"], row["Status"]))
    
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
        
    print("\n".join(output))
    return top_models

if __name__ == "__main__":
    audit_models()