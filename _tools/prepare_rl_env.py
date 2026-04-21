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
    RAW_DATA_FILE = BASE_DIR / "raw_combined.csv"
    
    # Новая изолированная структура для RL
    RL_ENV_DIR = BASE_DIR / "rl_env"
    CHAMPIONS_DIR = RL_ENV_DIR / "champions"
    OUTPUT_FILE = RL_ENV_DIR / "environment_data.parquet"
    METADATA_FILE = RL_ENV_DIR / "env_metadata.json"
    
    RL_ENV_DIR.mkdir(parents=True, exist_ok=True)
    CHAMPIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not RAW_DATA_FILE.exists():
        print(f"❌ Не найден базовый файл котировок {RAW_DATA_FILE}")
        return

    # Читаем глобальные метаданные (чтобы узнать lookback)
    with open(BASE_DIR / "metadata.json", 'r') as f:
        lookback = json.load(f)["parameters"]["lookback"]

    folds = sorted([d for d in BASE_DIR.glob("fold_*") if d.is_dir()])
    all_rl_data = []
    env_metadata = {"champions": {}}

    print(f"🚀 Старт сборки RL Environment (Macro F1-Score Casting)")
    print(f"📁 Целевая папка: {RL_ENV_DIR.absolute()}\n")

    for fold_dir in folds:
        fold_name = fold_dir.name
        models_dir = fold_dir / "models"
        val_parquet = fold_dir / "data" / "val" / "ml_data.parquet"
        features_json = fold_dir / "artifacts" / "features_selected.json"
        
        models = list(models_dir.glob("*.keras"))
        if not models or not val_parquet.exists():
            continue
            
        print(f"🎬 [{fold_name}] Аудит {len(models)} моделей-кандидатов...")
        
        # 1. Загружаем отобранные признаки для этого фолда
        with open(features_json, 'r') as f:
            feature_cols = json.load(f)["feature_order"]
            
        # 2. Загружаем полный валидационный датасет (все 149 фичей)
        val_df = pd.read_parquet(val_parquet)
        val_df['datetime'] = pd.to_datetime(val_df['datetime'])
        
        # 3. Нарезаем окна только из отобранных фичей (для LSTM)
        X_val, y_val, dates_val, tickers_val = create_sequences(val_df, feature_cols, lookback)
        
        if len(X_val) == 0:
            print(f"  ⚠️ Недостаточно данных для формирования окон. Пропуск.")
            continue

        best_f1 = -1.0
        best_model_probs = None
        best_model_path = None
        
        # 4. КАСТИНГ: Ищем модель с лучшим F1-Score
        for model_path in models:
            tf.keras.backend.clear_session() # Очистка VRAM перед загрузкой новой модели
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                # Быстрый прогон через GPU
                probs = model.predict(X_val, batch_size=2048, verbose=0)
                preds = np.argmax(probs, axis=1)
                
                # Считаем Macro F1-Score
                score = f1_score(y_val, preds, average='macro')
                
                if score > best_f1:
                    best_f1 = score
                    best_model_probs = probs
                    best_model_path = model_path
            except Exception as e:
                print(f"  ❌ Ошибка загрузки {model_path.name}: {e}")
                
        if best_model_path is None:
            continue
            
        print(f"  🏆 Чемпион: {best_model_path.name}")
        print(f"  📊 Macro F1: {best_f1*100:.2f}%")
        
        # 5. Сохраняем чемпиона в папку RL и пишем в метаданные
        shutil.copy2(best_model_path, CHAMPIONS_DIR / best_model_path.name)
        env_metadata["champions"][fold_name] = {
            "model_file": best_model_path.name,
            "macro_f1": float(best_f1),
            "features_used": len(feature_cols)
        }
        
        # 6. Создаем DataFrame с вероятностями
        preds_df = pd.DataFrame({
            'datetime': pd.to_datetime(dates_val),
            'ticker': tickers_val,
            'prob_SL': best_model_probs[:, 0],
            'prob_Hold': best_model_probs[:, 1],
            'prob_TP': best_model_probs[:, 2]
        })
        
        # 7. СЭНДВИЧ: Приклеиваем вероятности к полному датасету (со 149 фичами)
        # INNER JOIN автоматически отбросит первые `lookback` дней, для которых нет прогноза
        fold_rl_df = pd.merge(val_df, preds_df, on=['datetime', 'ticker'], how='inner')
        all_rl_data.append(fold_rl_df)

    if not all_rl_data:
        print("❌ Не найдено успешных предсказаний. Сборка остановлена.")
        return

    # 8. Склейка всех фолдов в непрерывный таймлайн
    print("\n🧩 Склеиваем Walk-Forward фолды...")
    rl_combined_df = pd.concat(all_rl_data, ignore_index=True)
    
    # 9. Обогащение сырыми котировками (для симулятора PnL)
    print("📥 Подтягиваем сырые OHLCV котировки...")
    raw_df = pd.read_csv(RAW_DATA_FILE)
    raw_df['datetime'] = pd.to_datetime(raw_df['datetime'])
    raw_df = raw_df[['datetime', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
    
    # Финальный джоин
    final_env_df = pd.merge(rl_combined_df, raw_df, on=['datetime', 'ticker'], how='inner')
    
    # Сортировка для красоты и порядка
    final_env_df = final_env_df.sort_values(['ticker', 'datetime']).reset_index(drop=True)

    # 10. Сохранение результатов
    final_env_df.to_parquet(OUTPUT_FILE, index=False)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(env_metadata, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 ГОТОВО! Песочница для RL-агента успешно собрана.")
    print(f"   Сохранено в: {OUTPUT_FILE.absolute()}")
    print(f"   Всего строк: {len(final_env_df)}")
    print(f"   Количество признаков (State Space): {len(final_env_df.columns)} колонок")

if __name__ == "__main__":
    main()