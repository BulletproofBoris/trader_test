import sys
import os
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent

def reduce_multicollinearity(df, feature_cols, threshold=0.85):
    print(f"📉 Удаление мультиколлинеарности (порог {threshold})...")
    sample_df = df.sample(n=min(len(df), 50000), random_state=42)
    corr_matrix = sample_df[feature_cols].corr(method='spearman').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    retained = [c for c in feature_cols if c not in to_drop]
    print(f"  ✅ Осталось признаков: {len(retained)} из {len(feature_cols)}")
    return retained

def get_dynamic_feature_importance(df, feature_cols, target_col='label', cumulative_threshold=0.95):
    print(f"🧠 LightGBM оценивает важность {len(feature_cols)} признаков...")
    sample_df = df.sample(n=min(len(df), 100000), random_state=42)
    X = sample_df[feature_cols]
    y = sample_df[target_col]
    
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X, y)
    
    importance = model.feature_importances_
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importance})
    imp_df = imp_df.sort_values('importance', ascending=False)
    
    imp_df['cumulative_importance'] = imp_df['importance'].cumsum() / imp_df['importance'].sum()
    selected_features = imp_df[imp_df['cumulative_importance'] <= cumulative_threshold]['feature'].tolist()
    
    if not selected_features:
        selected_features = imp_df['feature'].tolist()[:10]
        
    return selected_features, imp_df

def plot_importance(imp_df, top_n, save_path):
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=imp_df.head(top_n))
    plt.title(f'Топ-{top_n} признаков (LightGBM)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(args):
    EXP_DIR = BASE_DIR / "experiments" / args.exp_name
    ML_DATA_DIR = EXP_DIR / "dataset"
    MODELS_DIR = EXP_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_path = ML_DATA_DIR / "train_scaled.csv"
    if not train_path.exists():
        print(f"❌ Датасет не найден: {train_path}"); return

    print(f"🚀 Динамический отбор признаков для: {args.exp_name}")
    df = pd.read_csv(train_path)
    
    exclude_cols = {'datetime', 'ticker', 'label'}
    feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols]
    
    if not args.skip_corr:
        feature_cols = reduce_multicollinearity(df, feature_cols, threshold=args.corr_threshold)
        
    top_features, imp_df = get_dynamic_feature_importance(df, feature_cols, cumulative_threshold=args.cum_threshold)
    
    out_json = ML_DATA_DIR / "features_selected.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"feature_order": top_features}, f, indent=2)
        
    plot_importance(imp_df, len(top_features), MODELS_DIR / "dynamic_feature_importance.png")
    print(f"✅ Отбор завершен! Выбрано: {len(top_features)} признаков. Сохранено в {out_json.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--cum_threshold", type=float, default=0.95)
    parser.add_argument("--corr_threshold", type=float, default=0.85)
    parser.add_argument("--skip_corr", action="store_true")
    args = parser.parse_args()
    main(args)