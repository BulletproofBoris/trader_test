import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os

# --- Настройка путей ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "2_results"
FEATURES_DIR_PROCESSED = RESULTS_DIR / "features_processed"

def main(args):
    print("--- 🔄 Запуск объединения признаков ---")

    # --- 👇 ИЗМЕНЕНИЕ: Универсальная логика путей 👇 ---
    manual_dir = FEATURES_DIR_PROCESSED
    # Папка с новыми признаками (AE или PCA)
    new_features_dir = RESULTS_DIR / args.new_features_path
    output_dir = RESULTS_DIR / "features_combined" / args.output_model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    # --- 👆 КОНЕЦ ИЗМЕНЕНИЯ 👆 ---

    print(f"Источник ручных признаков: {manual_dir}")
    print(f"Источник новых признаков: {new_features_dir}")
    print(f"Результат будет сохранен в: {output_dir}")

    manual_files = list(manual_dir.glob("*.csv"))
    if not manual_files:
        print(f"❌ Ошибка: Не найдены файлы с ручными признаками в {manual_dir}")
        return

    for manual_file_path in tqdm(manual_files, desc="Объединение файлов"):
        base_filename = os.path.basename(manual_file_path)
        new_feature_file_path = new_features_dir / base_filename

        if not new_feature_file_path.exists():
            print(f"⚠️ Пропуск: для файла {base_filename} не найден соответствующий файл с новыми признаками.")
            continue

        df_manual = pd.read_csv(manual_file_path)
        df_new = pd.read_csv(new_feature_file_path)

        # Выбираем из df_new только новые признаки (AE_* или PCA_*) и ключ для слияния
        new_feature_cols = [col for col in df_new.columns if col.startswith(('AE_', 'PCA_'))]
        cols_to_merge = ['datetime'] + new_feature_cols
        df_to_merge = df_new[cols_to_merge].copy()

        # Объединяем по дате
        df_manual['datetime'] = pd.to_datetime(df_manual['datetime'])
        df_to_merge['datetime'] = pd.to_datetime(df_to_merge['datetime'])
        
        combined_df = pd.merge(df_manual, df_to_merge, on="datetime", how="left")
        
        combined_df.to_csv(output_dir / base_filename, index=False)

    print(f"\n✅ Процесс завершен. Объединенные признаки сохранены в: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine manual and generated (AE or PCA) features.")
    parser.add_argument("--output_model_name", type=str, required=True, help="A new, unique name for the combined feature set.")
    # --- 👇 ИЗМЕНЕНИЕ: Один универсальный аргумент вместо нескольких флагов 👇 ---
    parser.add_argument("--new_features_path", type=str, required=True, 
                        help="Path to the new features relative to '2_results' (e.g., 'features_pca/pca_v1' or 'features_encoded/dae_v1').")
    
    args = parser.parse_args()
    main(args)