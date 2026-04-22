import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

DATA_PATH = Path("data/processed/2000_2026_1d/rl_env/environment_data.parquet")
OUTPUT_CSV = Path("data/processed/2000_2026_1d/rl_env/lstm_evaluation_report.csv")

def evaluate_predictions():
    print(f"📊 Загрузка данных из {DATA_PATH}...")
    if not DATA_PATH.exists():
        print("❌ Файл не найден!")
        return

    df = pd.read_parquet(DATA_PATH)
    
    prob_cols = [c for c in df.columns if 'prob' in c.lower() or 'pred' in c.lower()]
    if len(prob_cols) < 3:
        print("❌ Не удалось найти 3 колонки вероятностей.")
        return

    print("⚙️ Расчет сигналов и доходностей...")
    # 0 = Short, 1 = Hold, 2 = Long
    df['pred_class'] = df[prob_cols[:3]].values.argmax(axis=1)
    
    # Переводим классы в торговые сигналы: -1 (шорт), 0 (кэш), 1 (лонг)
    df['signal'] = df['pred_class'].map({0: -1, 1: 0, 2: 1})
    
    price_col = 'close_y' if 'close_y' in df.columns else 'close'
    if price_col not in df.columns:
        price_col = [c for c in df.columns if 'close' in c.lower()][0]

    results = []
    
    # Группируем по тикерам
    tickers = df['ticker'].unique()
    print(f"🔍 Анализ {len(tickers)} тикеров...")
    
    for ticker in tickers:
        group = df[df['ticker'] == ticker].copy()
        group = group.sort_values('datetime').reset_index(drop=True)
        
        # Считаем доходность следующего дня
        group['fwd_return'] = group[price_col].pct_change().shift(-1)
        
        # Доходность стратегии: сигнал текущего дня умножить на доходность следующего
        group['strat_return'] = group['signal'] * group['fwd_return']
        
        # Статистика сигналов
        total_days = len(group)
        longs = (group['signal'] == 1).sum()
        shorts = (group['signal'] == -1).sum()
        holds = (group['signal'] == 0).sum()
        
        # Win Rate (доля прибыльных сделок среди активных дней)
        active_days = group[group['signal'] != 0].dropna(subset=['strat_return'])
        if len(active_days) > 0:
            win_rate = (active_days['strat_return'] > 0).mean() * 100
        else:
            win_rate = 0.0
            
        # Кумулятивная доходность (сложный процент)
        # Ограничим аномальные скачки (чтобы баги данных не ломали стату)
        group['fwd_return'] = group['fwd_return'].clip(-0.5, 0.5)
        group['strat_return'] = group['strat_return'].clip(-0.5, 0.5)
        
        cum_strat_return = (1 + group['strat_return'].fillna(0)).prod() - 1
        cum_bh_return = (1 + group['fwd_return'].fillna(0)).prod() - 1 # Buy & Hold
        
        results.append({
            'Ticker': ticker,
            'Total_Days': total_days,
            'Longs': longs,
            'Shorts': shorts,
            'Holds': holds,
            'Win_Rate_%': round(win_rate, 2),
            'Strat_Return_%': round(cum_strat_return * 100, 2),
            'B&H_Return_%': round(cum_bh_return * 100, 2),
            'Alpha_%': round((cum_strat_return - cum_bh_return) * 100, 2) # Превосходство над рынком
        })

    # Сохраняем отчет
    report_df = pd.DataFrame(results)
    
    # Сортируем по "Альфе" (превосходству над пассивным удержанием)
    report_df = report_df.sort_values('Alpha_%', ascending=False)
    report_df.to_csv(OUTPUT_CSV, index=False)
    
    # Выводим сводку
    print("\n" + "="*70)
    print("🏆 ТОП-5 ТИКЕРОВ ПО ПРЕВОСХОДСТВУ НАД РЫНКОМ (ALPHA)")
    print("="*70)
    print(report_df.head(5)[['Ticker', 'Win_Rate_%', 'Strat_Return_%', 'B&H_Return_%', 'Alpha_%']].to_string(index=False))
    
    print("\n" + "="*70)
    print("📉 ХУДШИЕ 5 ТИКЕРОВ ПО ПРЕВОСХОДСТВУ НАД РЫНКОМ (ALPHA)")
    print("="*70)
    print(report_df.tail(5)[['Ticker', 'Win_Rate_%', 'Strat_Return_%', 'B&H_Return_%', 'Alpha_%']].to_string(index=False))
    
    print("\n" + "="*70)
    print("📈 ОБЩАЯ СТАТИСТИКА ПО ВСЕМ ТИКЕРАМ")
    print("="*70)
    mean_winrate = report_df['Win_Rate_%'].mean()
    positive_alpha_count = (report_df['Alpha_%'] > 0).sum()
    print(f"Средний Win Rate: {mean_winrate:.2f}%")
    print(f"Тикеров, где LSTM обогнала Buy & Hold: {positive_alpha_count} из {len(tickers)} ({positive_alpha_count/len(tickers)*100:.1f}%)")
    
    print(f"\n✅ Полный отчет сохранен в: {OUTPUT_CSV}")

if __name__ == "__main__":
    evaluate_predictions()