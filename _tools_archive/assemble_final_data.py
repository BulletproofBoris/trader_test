# _tools/assemble_final_data.py
import argparse
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np  # для обработки inf/NaN векторно

# --- Настройка путей ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "2_results"

def load_file(file_path: Path) -> pd.DataFrame:
    """Загружает один CSV файл."""
    return pd.read_csv(file_path)

def main(args):
    """Собирает итоговый датасет для обучения модели."""
    print("--- ⚙️ Запуск сборки финального датасета ---")
    
    source_dir = RESULTS_DIR / args.source_dir_name
    features_json_path = RESULTS_DIR / f"selected_features_{args.model_name}.json"
    output_dir = RESULTS_DIR / "final_for_model" / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Источник файлов с признаками: {source_dir}")

    if not features_json_path.exists():
        print(f"❌ Ошибка: Файл с отобранными признаками не найден: {features_json_path}")
        return
        
    with open(features_json_path, 'r') as f:
        selected_features = json.load(f)
    print(f"Загружено {len(selected_features)} отобранных признаков.")

    all_files = sorted(list(source_dir.glob("*.csv")))
    if not all_files:
        print(f"❌ Ошибка: В исходной папке {source_dir} не найдено .csv файлов.")
        return

    print(f"Загрузка и объединение {len(all_files)} файлов в {args.workers} потоков...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(load_file, all_files), total=len(all_files)))
    
    full_df = pd.concat(results, ignore_index=True)

    essential_cols = ['datetime', 'ticker', 'label']
    final_cols = essential_cols + selected_features

    missing_cols = [col for col in final_cols if col not in full_df.columns]
    if missing_cols:
        print(f"❌ Ошибка: В исходных данных отсутствуют необходимые колонки: {missing_cols}")
        return

    final_df = full_df[final_cols].copy()

    print(f"✅ Сборка завершена. В итоговой таблице {len(final_df.columns)} колонок ({len(selected_features)} признаков).")

    final_df['datetime'] = pd.to_datetime(final_df['datetime'])
    final_df = final_df.sort_values(by="datetime").reset_index(drop=True)

    last_date = final_df['datetime'].max()
    split_point = last_date - pd.DateOffset(years=1)
    
    print(f"Точка разделения train/test: {split_point.date()}")
    
    train_df = final_df[final_df['datetime'] < split_point].copy()
    test_df  = final_df[final_df['datetime'] >= split_point].copy()

    # === УДАЛЕНИЕ NaN/±Inf в признаках БЕЗ утечки (медианы считаются по train) ===
    feature_cols = [c for c in selected_features if c not in ('datetime', 'ticker', 'label')]

    # 1) заменяем ±inf на NaN сразу во всех фичах (векторно, без chained assignment)
    for df_ in (train_df, test_df):
        df_.loc[:, feature_cols] = df_[feature_cols].replace([np.inf, -np.inf], np.nan)

    # 2) медианы считаем по TRAIN и только по числовым
    medians = train_df[feature_cols].median(numeric_only=True).fillna(0.0).to_dict()

    # диагностика до
    n_train_nan_before = int(train_df[feature_cols].isna().sum().sum())
    n_test_nan_before  = int(test_df[feature_cols].isna().sum().sum())

    # 3) заполняем NaN медианами ТРЕЙНа (векторно)
    train_df.loc[:, feature_cols] = train_df[feature_cols].fillna(medians)
    test_df.loc[:, feature_cols]  = test_df[feature_cols].fillna(medians)

    # диагностика после
    n_train_nan_after = int(train_df[feature_cols].isna().sum().sum())
    n_test_nan_after  = int(test_df[feature_cols].isna().sum().sum())

    print(f"Иммутация NaN завершена. NaN в train: {n_train_nan_before} → {n_train_nan_after}; "
          f"NaN в test: {n_test_nan_before} → {n_test_nan_after}")

    # === РЕМАП МЕТОК {-1,0,1} → {0,1,2} (оставляем convert_to_tfrecord без изменений) ===
    if 'label' in train_df.columns:
        train_df.loc[:, 'label'] = train_df['label'].map({-1: 0, 0: 1, 1: 2}).astype(int)
    if 'label' in test_df.columns:
        test_df.loc[:, 'label'] = test_df['label'].map({-1: 0, 0: 1, 1: 2}).astype(int)

    # === Сохранение ===
    train_output_path = output_dir / "train_final.csv"
    test_output_path  = output_dir / "test_final.csv"
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"\n✅ Обучающий набор сохранен в: {train_output_path}")
    print(f"✅ Тестовый набор сохранен в: {test_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble final dataset for model training.")
    parser.add_argument("--model_name", type=str, required=True, help="A name for the final dataset version.")
    parser.add_argument("--source_dir_name", type=str, required=True, 
                        help="Name of the source directory inside 2_results (e.g., 'features_combined/your_name').")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for data loading.")
    
    args = parser.parse_args()
    main(args)
