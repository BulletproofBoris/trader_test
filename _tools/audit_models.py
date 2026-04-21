import os
import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RL_DIR = BASE_DIR / "data" / "processed" / "2000_2026_1d" / "rl_env"
RESULTS_DIR = RL_DIR / "ray_results" / "pbt_trading_bot"
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

def audit_models():
    data = []
    print("🔍 Сканирование истории и наличия файлов чекпоинтов...")
    
    for root, dirs, files in os.walk(RESULTS_DIR):
        if "result.json" in files:
            folder_name = Path(root).name
            parts = folder_name.split("_")
            trial_id = "_".join(parts[2:4]) if len(parts) >= 4 else folder_name
            
            with open(os.path.join(root, "result.json"), "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        res = json.loads(line)
                        iteration = res.get("training_iteration", 0)
                        
                        train_ret = res.get("env_runners", {}).get("episode_return_mean")
                        eval_ret = res.get("evaluation", {}).get("env_runners", {}).get("episode_return_mean")
                        
                        if train_ret is not None and eval_ret is not None:
                            # ПРОВЕРКА НАЛИЧИЯ ФАЙЛА НА ДИСКЕ
                            checkpoint_name = f"checkpoint_{str(iteration).zfill(6)}"
                            full_path = os.path.join(root, checkpoint_name)
                            
                            if os.path.exists(full_path):
                                data.append({
                                    "Trial_ID": trial_id, 
                                    "Iteration": iteration,
                                    "Train_Ret": round(train_ret, 2), 
                                    "Test_Ret": round(eval_ret, 2), 
                                    "Path": root
                                })
                    except: pass

    if not data:
        print("❌ Не найдено ни одного сохраненного чекпоинта с метриками оценки.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["Status"] = df.apply(lambda row: classify_model(row["Train_Ret"], row["Test_Ret"]), axis=1)
    
    # Берем топ-15 из тех, что реально существуют
    top_models = df[df["Status"].str.contains("ПЕРСПЕКТИВНЫЙ")].sort_values("Test_Ret", ascending=False).head(15)

    if top_models.empty:
        # Если перспективных нет, давай возьмем топ-15 вообще любых существующих
        print("⚠️ 'Идеальных' моделей не найдено. Берем топ-15 доступных...")
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