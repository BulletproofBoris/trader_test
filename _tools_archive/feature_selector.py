import argparse
import json
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import catboost as cb
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss
import gc

# --- Настройка путей ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "2_results"

warnings.filterwarnings("ignore", category=UserWarning, module='catboost')

def load_features_from_files(source_dir: Path, n_files: int = None) -> pd.DataFrame:
    """Загружает признаки из CSV-файлов в указанной директории."""
    print(f"Загрузка файлов с признаками из: {source_dir}")
    files = sorted(list(source_dir.glob("*.csv")))
    if not files:
        raise FileNotFoundError(f"В директории {source_dir} не найдено CSV файлов.")
    
    if n_files:
        files = files[:n_files]
        
    df_list = [pd.read_csv(f) for f in tqdm(files, desc="Загрузка файлов с признаками")]
    return pd.concat(df_list, ignore_index=True)

def filter_by_correlation(df: pd.DataFrame, threshold: float) -> list:
    """Фильтрует признаки по порогу корреляции."""
    if threshold >= 1.0:
        print("Порог корреляции >= 1.0, фильтрация пропускается.")
        return df.columns.tolist()
        
    print(f"Фильтрация {df.shape[1]} признаков с порогом корреляции {threshold}...")
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    kept_features = df.drop(columns=to_drop).columns.tolist()
    print(f"Осталось {len(kept_features)} признаков после фильтрации.")
    return kept_features

def calculate_loss_importance_for_fold(model, X_val, y_val):
    """
    Рассчитывает Permutation Importance на основе logloss для одного фолда.
    """
    importances = {}
    baseline_preds = model.predict_proba(X_val)
    baseline_loss = log_loss(y_val, baseline_preds)

    for col in X_val.columns:
        X_val_permuted = X_val.copy()
        
        permuted_values = X_val_permuted[col].to_numpy()
        np.random.shuffle(permuted_values)
        X_val_permuted[col] = permuted_values
        
        permuted_preds = model.predict_proba(X_val_permuted)
        permuted_loss = log_loss(y_val, permuted_preds)
        
        importances[col] = permuted_loss - baseline_loss
        
    return importances

def main(args):
    source_dir = RESULTS_DIR / args.source_dir_name
    
    try:
        df = load_features_from_files(source_dir)
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}"); return

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    exclude_cols = {'datetime', 'ticker', 'label', 'meta_target', 'open', 'high', 'low', 'close', 'volume'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # --- 👇👇👇 ЕДИНСТВЕННОЕ ИЗМЕНЕНИЕ ЗДЕСЬ 👇👇👇 ---
    # Был исправлен синтаксис заполнения пропусков, чтобы убрать `FutureWarning` от pandas.
    # Логика осталась абсолютно идентичной.
    for col in tqdm(feature_cols, desc="Заполнение пропусков"):
        if df[col].isnull().any():
            # Старая строка: df[col].fillna(df[col].median(), inplace=True)
            df[col] = df[col].fillna(df[col].median())
    # --- 👆👆👆 КОНЕЦ ИЗМЕНЕНИЯ 👆👆👆 ---

    X = df[feature_cols]
    y = df['label']
    del df; gc.collect()
    
    all_importances = []
    
    print(f"\n--- 🚀 Запуск отбора признаков по метрике: '{args.ranking_metric.upper()}' ---")
    
    for r in range(args.repeats):
        print(f"\n--- Повтор [{r + 1}/{args.repeats}] ---")
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42 + r)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"  Фолд [{fold + 1}/{args.n_splits}]...")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = cb.CatBoostClassifier(
                iterations=500, learning_rate=0.05, depth=6,
                loss_function='MultiClass', eval_metric='Accuracy',
                random_seed=42, verbose=0, task_type="GPU" if args.use_gpu else "CPU",
                early_stopping_rounds=50
            )
            
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
            
            fold_importances = {}
            if args.ranking_metric == 'loss':
                fold_importances = calculate_loss_importance_for_fold(model, X_val, y_val)
            else: # 'accuracy'
                result = permutation_importance(
                    model, X_val, y_val, n_repeats=1, random_state=42, n_jobs=-1
                )
                for i, feat in enumerate(X_val.columns):
                    fold_importances[feat] = result.importances_mean[i]
            
            all_importances.append(fold_importances)
            del X_train, X_val, y_train, y_val, model; gc.collect()

    ranked_features = pd.DataFrame(all_importances).mean()
    
    if args.ranking_metric == 'loss':
        ranked_features.sort_values(ascending=True, inplace=True)
        print("\n--- Рейтинг признаков по ВЛИЯНИЮ НА LOSS (отрицательные = вредные) ---")
        print("   loss_impact < 0: Вредный признак (удаление улучшает/уменьшает loss)")
        print("   loss_impact > 0: Полезный признак (удаление ухудшает/увеличивает loss)")
    else: # 'accuracy'
        ranked_features.sort_values(ascending=False, inplace=True)
        print("\n--- Рейтинг признаков по ВЛИЯНИЮ НА ТОЧНОСТЬ (положительные = полезные) ---")

    kept_after_corr = filter_by_correlation(X[ranked_features.head(args.top_n).index], args.corr_threshold)
    final_feature_list = [feat for feat in ranked_features.index if feat in kept_after_corr]
    
    if len(final_feature_list) > args.top_n:
        final_feature_list = final_feature_list[:args.top_n]

    print(f"\n--- Итоговый рейтинг (топ-20) ---")
    print(ranked_features.head(20).to_string())

    output_path = RESULTS_DIR / f"selected_features_{args.model_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_feature_list, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Список из {len(final_feature_list)} признаков сохранен в: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select best features using CatBoost and correlation filtering.")
    parser.add_argument("--model_name", type=str, required=True, help="A unique name for the final selected feature set.")
    parser.add_argument("--source_dir_name", type=str, default="features_processed", 
                        help="Name of the source directory inside 2_results (e.g., 'features_processed' or 'features_combined/your_name').")
    parser.add_argument("--ranking_metric", type=str, default="accuracy", choices=["accuracy", "loss"], 
                        help="Metric to use for permutation importance: 'accuracy' (default) or 'loss'.")
    parser.add_argument("--top_n", type=int, default=160, help="Number of top features to consider for correlation filtering and final selection.")
    parser.add_argument("--corr_threshold", type=float, default=1.0, help="Correlation threshold to filter features.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat the cross-validation process for stable importance scores.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for StratifiedKFold cross-validation.")
    parser.add_argument("--use_gpu", action='store_true', help="Use GPU for CatBoost training.")
    
    args = parser.parse_args()
    main(args)