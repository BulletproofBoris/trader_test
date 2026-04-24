import os
import json
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def create_sequences(df, feature_cols, lookback):
    X, y, dates, tickers = [], [], [], []
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('datetime')
        features = group[feature_cols].values.astype(np.float32)
        labels = (group['label'].values + 1).astype(int) 
        dt = group['datetime'].values
        tk = group['ticker'].values
        
        for i in range(len(features) - lookback):
            X.append(features[i : i + lookback])
            y.append(labels[i + lookback - 1])
            dates.append(dt[i + lookback - 1])
            tickers.append(tk[i + lookback - 1])
            
    return np.array(X), np.array(y), np.array(dates), np.array(tickers)

def main():
    PROCESSED_DIR = Path("data/processed")
    
    # Список наших конфигураций-ансамблей
    CONFIGS = {
        "c60": "2000_2026_1d_60_10",
        "c30": "2000_2026_1d_30_5",
        "c18": "2000_2026_1d_18_3"
    }
    
    # Сохранять результат будем в общую родительскую папку
    RL_ENV_DIR = PROCESSED_DIR / "2000_2026_1d" / "rl_env"
    CHAMPIONS_DIR = RL_ENV_DIR / "champions"
    OUTPUT_FILE = RL_ENV_DIR / "environment_data.parquet"
    METADATA_FILE = RL_ENV_DIR / "env_metadata.json"
    
    # Очистка и создание папок
    if CHAMPIONS_DIR.exists():
        shutil.rmtree(CHAMPIONS_DIR)
    CHAMPIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Узнаем список фолдов (берем из первой конфиги)
    base_config_dir = PROCESSED_DIR / CONFIGS["c60"]
    if not base_config_dir.exists():
        print(f"❌ Ошибка: Не найдена базовая папка {base_config_dir}")
        return
        
    folds = sorted([d.name for d in base_config_dir.glob("fold_*") if d.is_dir()])
    
    all_rl_data = []
    env_metadata = {}

    print(f"🚀 СТАРТ: Сборка Мульти-Таймфрейм Ансамбля (9 нейросетей)")
    
    for fold_name in folds:
        print(f"\n🧩 Собираем консилиум для {fold_name}:")
        fold_merged_df = None
        base_features_df = None 
        env_metadata[fold_name] = {}

        # Проходим по каждой из 3-х конфигураций
        for conf_prefix, conf_folder in CONFIGS.items():
            conf_dir = PROCESSED_DIR / conf_folder
            fold_dir = conf_dir / fold_name
            
            print(f"  🔍 Анализ {conf_folder}...")
            
            meta_path = conf_dir / "metadata.json"
            if not meta_path.exists():
                print(f"    ❌ Пропуск: Нет metadata.json")
                continue
                
            with open(meta_path, 'r') as f:
                lookback = json.load(f)["parameters"]["lookback"]

            models_dir = fold_dir / "models"
            val_parquet = fold_dir / "data" / "val" / "ml_data.parquet"
            features_json = fold_dir / "artifacts" / "features_selected.json"
            
            if not models_dir.exists() or not val_parquet.exists():
                print(f"    ⚠️ Пропуск: Нет папки models или данных")
                continue
                
            with open(features_json, 'r') as f:
                feature_cols = json.load(f)["feature_order"]
                
            val_df = pd.read_parquet(val_parquet)
            val_df['datetime'] = pd.to_datetime(val_df['datetime'])
            
            # Фичи у всех конфигов одинаковые, берем их один раз из c60
            if base_features_df is None and conf_prefix == "c60":
                base_features_df = val_df.copy()
            
            X_val, y_val, dates_val, tickers_val = create_sequences(val_df, feature_cols, lookback)
            if len(X_val) == 0: continue

            models = list(models_dir.glob("*.keras"))
            model_scores = []
            
            # Прогон всех моделей конфига
            for model_path in models:
                tf.keras.backend.clear_session()
                try:
                    model = tf.keras.models.load_model(model_path, compile=False)
                    probs = model.predict(X_val, batch_size=2048, verbose=0)
                    preds = np.argmax(probs, axis=1)
                    score = f1_score(y_val, preds, average='macro')
                    model_scores.append({"path": model_path, "f1": score, "probs": probs})
                except Exception:
                    pass
                    
            # Берем ТОП-3
            model_scores.sort(key=lambda x: x["f1"], reverse=True)
            top_models = model_scores[:3]
            if not top_models: continue
                
            preds_dict = {'datetime': pd.to_datetime(dates_val), 'ticker': tickers_val}
            env_metadata[fold_name][conf_prefix] = {}
            
            # Записываем вероятности и сохраняем чемпионов
            for i, m_info in enumerate(top_models, 1):
                model_name = m_info['path'].name
                new_name = f"{conf_prefix}_rank{i}_{model_name}" # Пример: c60_rank1_model.keras
                
                print(f"    🏆 {new_name} (F1: {m_info['f1']*100:.1f}%)")
                shutil.copy2(m_info['path'], CHAMPIONS_DIR / new_name)
                
                env_metadata[fold_name][conf_prefix][f"rank_{i}"] = {"model_file": new_name}
                
                # Колонки будут: c60_m1_p0, c18_m3_p2 и т.д.
                preds_dict[f'{conf_prefix}_m{i}_p0'] = m_info['probs'][:, 0]
                preds_dict[f'{conf_prefix}_m{i}_p1'] = m_info['probs'][:, 1]
                preds_dict[f'{conf_prefix}_m{i}_p2'] = m_info['probs'][:, 2]

            # Сливаем (Inner Join) предсказания текущего конфига с остальными
            preds_df = pd.DataFrame(preds_dict)
            if fold_merged_df is None:
                fold_merged_df = preds_df
            else:
                fold_merged_df = pd.merge(fold_merged_df, preds_df, on=['datetime', 'ticker'], how='inner')
        
        # Если удалось собрать данные хоть с кого-то
        if fold_merged_df is not None and not fold_merged_df.empty:
            # 1. Склеиваем с базовыми фичами
            fold_rl_df = pd.merge(base_features_df, fold_merged_df, on=['datetime', 'ticker'], how='inner')
            
            # 2. Подтягиваем сырые OHLCV котировки из папки c60
            val_csv = PROCESSED_DIR / CONFIGS["c60"] / fold_name / "data" / "val" / "dataset.csv"
            if val_csv.exists():
                raw_fold_df = pd.read_csv(val_csv)[['datetime', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
                raw_fold_df['datetime'] = pd.to_datetime(raw_fold_df['datetime'])
                fold_rl_df = pd.merge(fold_rl_df, raw_fold_df, on=['datetime', 'ticker'], how='inner')
                all_rl_data.append(fold_rl_df)

    if not all_rl_data:
        print("\n❌ Не удалось собрать данные. Проверь наличие обученных моделей во всех папках.")
        return

    print("\n🧩 Склеиваем Walk-Forward фолды...")
    final_env_df = pd.concat(all_rl_data, ignore_index=True)
    
    if 'close_y' not in final_env_df.columns and 'close' in final_env_df.columns:
        final_env_df['close_y'] = final_env_df['close']
        
    final_env_df = final_env_df.sort_values(['ticker', 'datetime']).reset_index(drop=True)
    final_env_df.to_parquet(OUTPUT_FILE, index=False)
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(env_metadata, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 ПЕСОЧНИЦА УСПЕШНО СОБРАНА!")
    print(f"   Файл: {OUTPUT_FILE}")
    print(f"   Количество строк: {len(final_env_df)}")
    print(f"   Количество фичей + вероятностей: {len(final_env_df.columns)}")

if __name__ == "__main__":
    main()