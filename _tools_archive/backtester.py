# _tools/backtester.py
import argparse
import json
import joblib
from pathlib import Path
import warnings
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.style as style
from tqdm import tqdm

# --- Настройка окружения и путей ---
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
style.use('dark_background')

# Импорты из вашего проекта
try:
    from _core.paths import (
        PROJECT_DIR, RESULTS_DIR, MODELS_DIR, SCALERS_DIR, 
        FEATURES_DIR_PROCESSED as FEATURES_DIR
    )
except ImportError:
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    RESULTS_DIR = PROJECT_DIR / "2_results"
    MODELS_DIR = RESULTS_DIR / "models"
    SCALERS_DIR = RESULTS_DIR / "scalers"
    FEATURES_DIR = RESULTS_DIR / "features_processed"

def create_sequences(data: np.ndarray, lookback: int):
    """Создает временные последовательности (сэмплы) для LSTM/GRU."""
    X = []
    for i in range(len(data) - lookback + 1):
        X.append(data[i:i+lookback])
    return np.array(X)

def main(args):
    print(f"--- 📈 Запуск бектеста для модели '{args.model_name}' на тикере '{args.ticker}' ---")

    # --- 1. Загрузка артефактов ---
    model_path = MODELS_DIR / args.model_name / "model.keras"
    scaler_path = SCALERS_DIR / f"{args.model_name}_scaler.pkl"
    features_path = SCALERS_DIR / f"{args.model_name}_features.json"
    metadata_path = RESULTS_DIR / f"tfrecord_{args.model_name}" / "metadata.json"

    try:
        print("Загрузка артефактов...")
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        with open(features_path, 'r') as f:
            feature_cols = json.load(f)['feature_order']
        with open(metadata_path, 'r') as f:
            lookback = json.load(f)['lookback']
        print(f"✅ Артефакты загружены. Lookback: {lookback}, Признаков: {len(feature_cols)}")
    except FileNotFoundError as e:
        print(f"❌ Ошибка: Не найден один из необходимых артефактов. {e}")
        return

    # --- 2. Загрузка и подготовка данных ---
    data_file = FEATURES_DIR / f"{args.ticker}.csv"
    if not data_file.exists():
        print(f"❌ Ошибка: Не найден файл с признаками для тикера: {data_file}")
        return

    print(f"Загрузка данных для тикера из {data_file}...")
    df = pd.read_csv(data_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    end_date = df['datetime'].max()
    start_date = end_date - pd.DateOffset(years=args.years)
    df_backtest = df[df['datetime'] >= start_date].copy()
    
    if len(df_backtest) < lookback:
        print(f"❌ Ошибка: Недостаточно данных для бектеста ({len(df_backtest)} строк), требуется минимум {lookback}.")
        return

    print(f"Подготовка {len(df_backtest)} строк данных для модели...")
    X_raw = df_backtest[feature_cols].values
    X_scaled = scaler.transform(X_raw)
    X_sequences = create_sequences(X_scaled, lookback)

    # --- 3. Получение предсказаний ---
    print("Получение предсказаний модели...")
    predictions_proba = model.predict(X_sequences, verbose=0)
    predictions = np.argmax(predictions_proba, axis=1)
    
    prediction_dates = df_backtest['datetime'].iloc[lookback-1:].reset_index(drop=True)
    df_signals = pd.DataFrame({'datetime': prediction_dates, 'signal': predictions})
    df_backtest = pd.merge(df_backtest, df_signals, on='datetime', how='left')

    # --- 4. Визуализация (ИСПРАВЛЕННАЯ ЛОГИКА) ---
    print("Создание графика...")

    # --- 👇👇👇 НОВОЕ РЕШЕНИЕ 👇👇👇 ---
    # 1. Создаем копию колонки 'close' для отрисовки.
    df_backtest['plot_close'] = df_backtest['close']
    # 2. Находим неторговые дни (где объем равен 0 или NaN) и ставим в `plot_close` значение NaN.
    #    Matplotlib не будет соединять линиями точки, между которыми есть NaN.
    df_backtest.loc[df_backtest['volume'].fillna(0) == 0, 'plot_close'] = np.nan
    # --- 👆👆👆 КОНЕЦ НОВОГО РЕШЕНИЯ 👆👆👆 ---

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Верхний график: Цена и сигналы
    ax1.plot(df_backtest['datetime'], df_backtest['plot_close'], label='Цена Close', color='cyan', alpha=0.9, linewidth=1.2)
    
    # Для сигналов используем оригинальный df_backtest, чтобы точки рисовались на правильной высоте
    buy_signals = df_backtest[df_backtest['signal'] == 2]
    sell_signals = df_backtest[df_backtest['signal'] == 0]
    
    ax1.scatter(buy_signals['datetime'], buy_signals['close'], label='Buy', marker='^', color='lime', s=120, zorder=5, edgecolors='black', linewidths=0.5)
    ax1.scatter(sell_signals['datetime'], sell_signals['close'], label='Sell', marker='v', color='red', s=120, zorder=5, edgecolors='black', linewidths=0.5)
    
    ax1.set_title(f"Бектест модели '{args.model_name}' на '{args.ticker}' за последний год", fontsize=16)
    ax1.set_ylabel("Цена", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend()
    
    # Нижний график (без изменений)
    ax2.plot(df_backtest['datetime'], np.ones(len(df_backtest)), color='gray', linestyle='--')
    ax2.set_title("Portfolio Value (Work in Progress)", fontsize=12)
    ax2.set_xlabel("Дата", fontsize=12)
    ax2.set_ylabel("Value", fontsize=12)
    
    plt.tight_layout()
    
    report_dir = RESULTS_DIR / "reports"
    report_dir.mkdir(exist_ok=True, parents=True)
    output_path = report_dir / f"backtest_{args.model_name}_{args.ticker}.png"
    plt.savefig(output_path)
    print(f"✅ График бектеста сохранен в: {output_path}")

    # --- 5. Простая статистика ---
    trade_days = df_backtest[df_backtest['volume'].fillna(0) > 0]
    n_buys = len(buy_signals)
    n_sells = len(sell_signals)
    buy_and_hold_return = (trade_days['close'].iloc[-1] / trade_days['close'].iloc[0] - 1) * 100
    
    print("\n--- 📊 Базовая статистика ---")
    print(f"Количество сигналов 'Buy': {n_buys}")
    print(f"Количество сигналов 'Sell': {n_sells}")
    print(f"Доходность 'Купи и держи' за период: {buy_and_hold_return:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск бектеста обученной модели на одном тикере.")
    parser.add_argument("--model_name", type=str, required=True, help="Имя модели для тестирования.")
    parser.add_argument("--ticker", type=str, required=True, help="Тикер акции для бектеста (напр. SBER).")
    parser.add_argument("--years", type=int, default=1, help="Количество лет для периода бектеста.")
    
    args = parser.parse_args()
    main(args)