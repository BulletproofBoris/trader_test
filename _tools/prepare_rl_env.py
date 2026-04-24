import os
import json
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import f1_score

# Отключаем системный спам TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def create_sequences(df, feature_cols, lookback=60):
    """
    Нарезает датафрейм на окна [lookback, features] для LSTM.
    Возвращает тензоры, а также списки дат и тикеров для точной склейки.
    """
    X, y, dates, tickers = [], [], [], []
    
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('datetime')
        features = group[feature_cols].values.astype(np.float32)
        
        # Сдвигаем метки: -1 (SL) -> 0, 0 (Hold) -> 1, 1 (TP) -> 2
        labels = (group['label'].values + 1).astype(int) 
        
        dt = group['datetime'].values
        tk = group['ticker'].values
        
        for i in range(len(features) - lookback):
            X.append(features[i : i + lookback])
            y.append(labels[i + lookback - 1])
            # Привязываем прогноз к последнему дню окна
            dates.append(dt[i + lookback - 1])
            tickers.append(tk[i + lookback - 1])
            
    return np.array(X), np.array(y), np.array(dates), np.array(tickers)

def main():
    BASE_DIR = Path("data/processed/2000_2026_1d")
    
    # Изолированная структура для RL
    RL_ENV_DIR = BASE_DIR / "rl_env"
    CHAMPIONS_DIR = RL_ENV_DIR / "champions"
    OUTPUT_FILE = RL_ENV_DIR / "environment_data.parquet"
    METADATA_FILE = RL_ENV_DIR / "env_metadata.json"
    
    RL_ENV_DIR.mkdir(parents=True, exist_ok=True)
    CHAMPIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Читаем глобальные метаданные (чтобы узнать lookback)
    meta_path = BASE_DIR / "metadata.json"
    if not meta_path.exists():
        print(f"❌ Не найден файл {meta_path}. Проверь путь к датасету.")
        return
        
    with open(meta_path, 'r') as f:
        lookback = json.load(f)["parameters"]["lookback"]

    folds = sorted([d for d in BASE_DIR.glob("fold_*") if d.is_dir()])
    all_rl_data = []
    env_metadata = {"champions": {}}

    print(f"🚀 Старт сборки RL Environment (Multi-Model F1-Score Casting)")
    print(f"📁 Целевая папка: {RL_ENV_DIR.absolute()}\n")

    for fold_dir in folds:
        fold_name = fold_dir.name
        models_dir = fold_dir / "models"
        val_parquet = fold_dir / "data" / "val" / "ml_data.parquet"
        val_csv = fold_dir / "data" / "val" / "dataset.csv"  # <-- Берем сырые данные отсюда!
        features_json = fold_dir / "artifacts" / "features_selected.json"
        
        models = list(models_dir.glob("*.keras"))
        if not models or not val_parquet.exists() or not val_csv.exists():
            continue
            
        print(f"🎬 [{fold_name}] Аудит {len(models)} моделей-кандидатов...")
        
        with open(features_json, 'r') as f:
            feature_cols = json.load(f)["feature_order"]
            
        val_df = pd.read_parquet(val_parquet)
        val_df['datetime'] = pd.to_datetime(val_df['datetime'])
        
        X_val, y_val, dates_val, tickers_val = create_sequences(val_df, feature_cols, lookback)
        
        if len(X_val) == 0:
            print(f"  ⚠️ Недостаточно данных для формирования окон. Пропуск.")
            continue

        model_scores = []
        
        for model_path in models:
            tf.keras.backend.clear_session()
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                probs = model.predict(X_val, batch_size=2048, verbose=0)
                preds = np.argmax(probs, axis=1)
                
                score = f1_score(y_val, preds, average='macro')
                
                model_scores.append({
                    "path": model_path,
                    "f1": score,
                    "probs": probs
                })
            except Exception as e:
                print(f"  ❌ Ошибка загрузки {model_path.name}: {e}")
                
        model_scores.sort(key=lambda x: x["f1"], reverse=True)
        top_models = model_scores[:3]

        if len(top_models) < 3:
            print("  ⚠️ Найдено менее 3-х успешных моделей. Пропуск фолда.")
            continue
            
        env_metadata["champions"][fold_name] = {}
        
        preds_dict = {
            'datetime': pd.to_datetime(dates_val),
            'ticker': tickers_val
        }
        
        for i, m_info in enumerate(top_models, 1):
            model_name = m_info['path'].name
            print(f"  🏆 Чемпион {i}: {model_name} (Macro F1: {m_info['f1']*100:.2f}%)")
            
            shutil.copy2(m_info['path'], CHAMPIONS_DIR / f"rank{i}_{model_name}")
            
            env_metadata["champions"][fold_name][f"rank_{i}"] = {
                "model_file": model_name,
                "macro_f1": float(m_info['f1']),
                "features_used": len(feature_cols)
            }
            
            preds_dict[f'm{i}_p0'] = m_info['probs'][:, 0]
            preds_dict[f'm{i}_p1'] = m_info['probs'][:, 1]
            preds_dict[f'm{i}_p2'] = m_info['probs'][:, 2]

        preds_df = pd.DataFrame(preds_dict)
        
        # Приклеиваем вероятности к ml_data (фичам)
        fold_rl_df = pd.merge(val_df, preds_df, on=['datetime', 'ticker'], how='inner')
        
        # Подтягиваем сырые OHLCV котировки прямо из папки фолда
        raw_fold_df = pd.read_csv(val_csv)[['datetime', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
        raw_fold_df['datetime'] = pd.to_datetime(raw_fold_df['datetime'])
        
        # Приклеиваем сырые котировки (Pandas сам добавит суффиксы _x и _y, если есть дубликаты имен)
        fold_rl_df = pd.merge(fold_rl_df, raw_fold_df, on=['datetime', 'ticker'], how='inner')
        
        all_rl_data.append(fold_rl_df)

    if not all_rl_data:
        print("❌ Не найдено успешных предсказаний. Сборка остановлена.")
        return

    print("\n🧩 Склеиваем Walk-Forward фолды...")
    final_env_df = pd.concat(all_rl_data, ignore_index=True)
    
    # Гарантируем, что колонка close_y существует (этого ждет твой rl_env.py)
    if 'close_y' not in final_env_df.columns:
        if 'close' in final_env_df.columns:
            final_env_df['close_y'] = final_env_df['close']
        else:
            print("⚠️ ВНИМАНИЕ: Колонка с ценой закрытия не найдена!")
    
    final_env_df = final_env_df.sort_values(['ticker', 'datetime']).reset_index(drop=True)

    final_env_df.to_parquet(OUTPUT_FILE, index=False)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(env_metadata, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 ГОТОВО! Песочница (Топ-3) для RL-агента успешно собрана.")
    print(f"   Сохранено в: {OUTPUT_FILE.absolute()}")
    print(f"   Всего строк: {len(final_env_df)}")
    print(f"   Количество признаков (State Space): {len(final_env_df.columns)} колонок")

if __name__ == "__main__":
    main()