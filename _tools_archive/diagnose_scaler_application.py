# _tools/diagnose_scaler_application.py
import argparse
import json
import joblib
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline

# --- Настройка путей ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "2_results"

def main(args):
    model_name = args.model_name
    print("--- 🩺 Диагностика применения скейлера ---")
    print(f"Модель: {model_name}")

    # --- Шаг 1: Загрузка артефактов ---
    try:
        scaler_dir = RESULTS_DIR / "scalers"
        scaler_path = scaler_dir / f"{model_name}_scaler.pkl"
        features_json_path = scaler_dir / f"{model_name}_features.json"

        print(f"Загружаю скейлер из: {scaler_path}")
        scaler: Pipeline = joblib.load(scaler_path)
        
        print(f"Загружаю список признаков из: {features_json_path}")
        with open(features_json_path, 'r') as f:
            selected_features = json.load(f)['feature_order']
        
        print(f"✅ Артефакты успешно загружены. Признаков для масштабирования: {len(selected_features)}")
    except Exception as e:
        print(f"❌ Ошибка на этапе загрузки артефактов: {e}")
        return

    # --- Шаг 2: Загрузка данных ---
    try:
        source_dir = RESULTS_DIR / "final_for_model" / model_name
        train_path = source_dir / "train_final.csv"
        print(f"Загружаю данные из: {train_path}")
        train_df = pd.read_csv(train_path)
        print(f"✅ Данные загружены. Форма: {train_df.shape}")
    except Exception as e:
        print(f"❌ Ошибка на этапе загрузки данных: {e}")
        return

    # --- Шаг 3: Применение скейлера ---
    try:
        print("\\n--- 🚀 Начинаю применение scaler.transform() к данным... ---")
        
        # Убедимся, что все колонки на месте
        for col in selected_features:
            if col not in train_df.columns:
                train_df[col] = 0.0
        
        # Выполняем трансформацию
        scaled_features = scaler.transform(train_df[selected_features])
        
        print("✅ scaler.transform() успешно выполнен!")
        
        # --- Шаг 4: Проверка результата ---
        print("\\n--- 📊 Анализ результата масштабирования ---")
        df_scaled = pd.DataFrame(scaled_features, columns=selected_features)
        
        stats = df_scaled.describe().transpose()[['mean', 'std', 'min', 'max']]
        print(stats.to_string())
        
        # Проверяем, что средние значения близки к нулю, а std - нет
        if stats['mean'].abs().max() < 0.1 and stats['std'].min() > 0.1:
            print("\\n🎉🎉🎉 ВЫВОД: Скейлер работает корректно! Данные успешно масштабированы. 🎉🎉🎉")
        else:
            print("\\n⚠️ ВНИМАНИЕ: Результат масштабирования выглядит подозрительно. Средние не центрированы или std слишком малы.")

    except Exception as e:
        print(f"❌ Ошибка на этапе применения scaler.transform(): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Диагностика применения обученного скейлера.")
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    main(args)