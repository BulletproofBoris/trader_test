import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def main():
    data_path = Path("data/processed/2000_2026_1d/rl_env/environment_data.parquet")
    output_dir = Path("data/processed/2000_2026_1d/rl_env/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Загрузка данных из {data_path}...")
    if not data_path.exists():
        print("❌ Файл environment_data.parquet не найден!")
        return

    df = pd.read_parquet(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Ищем колонку цены (обычно 'close')
    price_col = 'close_y' if 'close_y' in df.columns else 'close'
    if price_col not in df.columns:
        price_col = [c for c in df.columns if 'close' in c.lower()][0]

    # Находим самый часто встречающийся тикер (например, Сбербанк или индекс)
    top_ticker = df['ticker'].value_counts().index[0]
    print(f"🔍 Выбран тикер для анализа: {top_ticker}")

    # Фильтруем данные и берем последние 300 дней для красивой визуализации
    tdf = df[df['ticker'] == top_ticker].sort_values('datetime').reset_index(drop=True)
    
    # Защита от слишком коротких рядов
    plot_days = min(300, len(tdf))
    tdf = tdf.iloc[-plot_days:].reset_index(drop=True)
    
    print(f"📅 Отрисовка последних {plot_days} торговых дней (с {tdf['datetime'].iloc[0].date()} по {tdf['datetime'].iloc[-1].date()})")

    # --- ЛОГИКА СОГЛАСИЯ АНСАМБЛЯ (AGREEMENT) ---
    threshold = 0.40 # Порог уверенности
    
    # Условие: Все 3 модели дают вероятность Long > threshold
    tdf['agree_long'] = (tdf['m1_p2'] > threshold) & (tdf['m2_p2'] > threshold) & (tdf['m3_p2'] > threshold)
    # Условие: Все 3 модели дают вероятность Short > threshold
    tdf['agree_short'] = (tdf['m1_p0'] > threshold) & (tdf['m2_p0'] > threshold) & (tdf['m3_p0'] > threshold)

    agree_rate = (tdf['agree_long'].sum() + tdf['agree_short'].sum()) / len(tdf) * 100
    print(f"🤝 Процент полного согласия моделей (Agreement Rate): {agree_rate:.1f}%")

    # --- ВИЗУАЛИЗАЦИЯ ---
    fig, axs = plt.subplots(4, 1, figsize=(16, 14), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    fig.suptitle(f'Ансамбль LSTM (3 модели) | Тикер: {top_ticker} | Порог уверенности: {threshold}', fontsize=16, fontweight='bold')

    # 1. ГЛАВНЫЙ ГРАФИК ЦЕНЫ
    ax = axs[0]
    ax.plot(tdf['datetime'], tdf[price_col], label='Цена Close', color='black', linewidth=1.5)
    
    # Подсветка фона, когда модели согласны
    ax.fill_between(tdf['datetime'], tdf[price_col].min(), tdf[price_col].max(), 
                    where=tdf['agree_long'], facecolor='green', alpha=0.2, label='Все 3 за LONG')
    ax.fill_between(tdf['datetime'], tdf[price_col].min(), tdf[price_col].max(), 
                    where=tdf['agree_short'], facecolor='red', alpha=0.2, label='Все 3 за SHORT')

    ax.set_title('График цены и Зоны Согласия (Consensus Zones)', fontsize=12)
    ax.set_ylabel('Цена')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)

    # 2. ПУЛЬС МОДЕЛЕЙ (Функция для отрисовки)
    def plot_model_probs(ax, m_idx, m_name):
        ax.plot(tdf['datetime'], tdf[f'm{m_idx}_p2'], label='P(Long)', color='green', linewidth=1)
        ax.plot(tdf['datetime'], tdf[f'm{m_idx}_p0'], label='P(Short)', color='red', linewidth=1)
        ax.axhline(threshold, color='black', linestyle=':', alpha=0.5, label='Порог (Threshold)')
        
        ax.set_title(f'{m_name}', fontsize=10, loc='left')
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Вероятность')
        ax.legend(loc='upper left', fontsize=8, ncol=3)
        ax.grid(True, linestyle='--', alpha=0.5)

    plot_model_probs(axs[1], 1, "Модель 1 (Самый низкий Loss)")
    plot_model_probs(axs[2], 2, "Модель 2")
    plot_model_probs(axs[3], 3, "Модель 3")

    # Форматирование оси X (Даты)
    axs[3].xaxis.set_major_locator(mdates.MonthLocator())
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    
    # Сохранение
    plot_path = output_dir / f"{top_ticker.replace('@', '_')}_multimodel_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ График успешно сохранен: {plot_path}")

if __name__ == "__main__":
    main()