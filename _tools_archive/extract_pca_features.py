import argparse
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import json

# --- Настройка путей ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "2_results"
FEATURES_DIR_PROCESSED = RESULTS_DIR / "features_processed"

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Определяет список колонок, которые являются признаками."""
    exclude_cols = {'datetime', 'ticker', 'label', 'meta_target', 'open', 'high', 'low', 'close', 'volume'}
    return [col for col in df.columns if col not in exclude_cols]

def main(args):
    print("--- 🔬 Запуск извлечения ортогональных признаков (PCA) ---")
    
    output_dir = RESULTS_DIR / "features_pca" / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = output_dir / "scaler.pkl"
    pca_path = output_dir / "pca.pkl"
    metadata_path = output_dir / "metadata.json"
    print(f"Артефакты и новые признаки будут сохранены в: {output_dir}")

    print("Загрузка и объединение файлов с признаками...")
    all_files = list(FEATURES_DIR_PROCESSED.glob("*.csv"))
    if not all_files:
        print(f"❌ Ошибка: В папке {FEATURES_DIR_PROCESSED} не найдено файлов.")
        return

    df_list = [pd.read_csv(f) for f in tqdm(all_files)]
    full_df = pd.concat(df_list, ignore_index=True)
    full_df['datetime'] = pd.to_datetime(full_df['datetime'])
    full_df.sort_values(by=['ticker', 'datetime'], inplace=True)
    
    split_date = full_df['datetime'].max() - pd.DateOffset(years=1)
    train_df = full_df[full_df['datetime'] < split_date]
    
    feature_cols = get_feature_columns(train_df)
    print(f"Найдено {len(feature_cols)} исходных признаков для PCA.")
    
    X_train = train_df[feature_cols].values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print(f"Обучение PCA для извлечения {args.n_components} главных компонент...")
    pca = PCA(n_components=args.n_components)
    pca.fit(X_train_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"Всего {args.n_components} компонент объясняют: {np.sum(explained_variance):.2%} всей дисперсии.")

    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)
    print(f"✅ Скейлер и PCA обучены и сохранены.")

    # --- 👇 ИЗМЕНЕНИЕ: Применяем PCA и сохраняем по файлам, а не в один большой 👇 ---
    print("Применение PCA и сохранение по файлам...")
    pca_feature_names = [f"PCA_{i}" for i in range(args.n_components)]
    
    for ticker, group_df in tqdm(full_df.groupby('ticker'), desc="Обработка и сохранение тикеров"):
        meta_cols = [col for col in ['datetime', 'ticker', 'label'] if col in group_df.columns]
        df_meta = group_df[meta_cols].reset_index(drop=True)

        # Убедимся, что все фичи на месте перед трансформацией
        X_group = group_df[feature_cols].copy()
        missing_features = set(feature_cols) - set(X_group.columns)
        if missing_features:
            for feature in missing_features:
                X_group[feature] = 0
        X_group = X_group[feature_cols]

        X_group_np = np.nan_to_num(X_group.values, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_group_scaled = scaler.transform(X_group_np)
        X_pca = pca.transform(X_group_scaled)
        
        df_pca = pd.DataFrame(X_pca, columns=pca_feature_names)
        
        final_df = pd.concat([df_meta, df_pca], axis=1)
        
        # Сохраняем индивидуальный файл
        output_file_path = output_dir / f"{ticker}.csv"
        final_df.to_csv(output_file_path, index=False)
    # --- 👆 КОНЕЦ ИЗМЕНЕНИЯ 👆 ---
        
    metadata = { "model_name": args.model_name, "type": "PCA", "n_components": args.n_components, "explained_variance_ratio": explained_variance.tolist() }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"\n✅ Финальные наборы из {args.n_components} ортогональных признаков сохранены в: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract orthogonal features using PCA.")
    parser.add_argument("--model_name", type=str, required=True, help="A unique name for this PCA model/run.")
    parser.add_argument("--n_components", type=int, default=50, help="Number of principal components to extract.")
    args = parser.parse_args()
    main(args)