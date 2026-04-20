# _tools/apply_scaler.py
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
    print(f"---  Applying Scaler for model: {model_name} ---")

    # --- Входные пути ---
    source_dir = RESULTS_DIR / "final_for_model" / model_name
    scaler_dir = RESULTS_DIR / "scalers"
    scaler_path = scaler_dir / f"{model_name}_scaler.pkl"
    features_path = scaler_dir / f"{model_name}_features.json"
    train_in_path = source_dir / "train_final.csv"
    test_in_path = source_dir / "test_final.csv"

    # --- Выходные пути ---
    output_dir = RESULTS_DIR / "features_scaled" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    train_out_path = output_dir / "train_scaled.csv"
    test_out_path = output_dir / "test_scaled.csv"

    # --- Загрузка артефактов и данных ---
    print("Loading artifacts and data...")
    if not all(p.exists() for p in [scaler_path, features_path, train_in_path, test_in_path]):
        raise FileNotFoundError("One or more input files not found. Run assemble_final_data and fit_scaler first.")

    scaler: Pipeline = joblib.load(scaler_path)
    with open(features_path, 'r') as f:
        feature_cols = json.load(f)['feature_order']
    
    train_df = pd.read_csv(train_in_path)
    test_df = pd.read_csv(test_in_path)

    # --- Применение скейлера ---
    print("Applying scaler transformation...")
    
    # Сохраняем не-фичевые колонки
    train_meta_cols = train_df.drop(columns=feature_cols, errors='ignore')
    test_meta_cols = test_df.drop(columns=feature_cols, errors='ignore')

    # Применяем скейлер
    train_scaled = scaler.transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])

    # Собираем обратно в DataFrame
    train_df_scaled = pd.DataFrame(train_scaled, columns=feature_cols, index=train_meta_cols.index)
    train_df_final = pd.concat([train_meta_cols, train_df_scaled], axis=1)

    test_df_scaled = pd.DataFrame(test_scaled, columns=feature_cols, index=test_meta_cols.index)
    test_df_final = pd.concat([test_meta_cols, test_df_scaled], axis=1)
    
    # --- Сохранение ---
    print("Saving scaled data...")
    train_df_final.to_csv(train_out_path, index=False)
    test_df_final.to_csv(test_out_path, index=False)
    
    print(f"✅ Scaled data saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a pre-trained scaler to the final dataset.")
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    main(args)