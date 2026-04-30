import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import os

def decay_model(x, a, b, c):
    return a * np.exp(-b * x) + c

def main():
    parser = argparse.ArgumentParser(description="Анализатор гиперпараметрического поиска.")
    parser.add_argument("fold_dir", type=str, help="Путь к папке фолда")
    parser.add_argument("--runs", type=int, default=100, help="Запланированное количество ранов для прогноза")
    parser.add_argument("--max_x", type=int, default=None, help="Ограничение по оси X (раны)")
    parser.add_argument("--max_y", type=float, default=None, help="Жесткое ограничение по оси Y")
    
    # Теперь границы по умолчанию None (Включается режим АВТО)
    parser.add_argument("--valid_min", type=float, default=None, help="Нижняя граница (Ручная)")
    parser.add_argument("--valid_max", type=float, default=None, help="Верхняя граница (Ручная)")
    
    args = parser.parse_args()
    
    db_path = os.path.join(args.fold_dir, "trading_factory.db")
    target_runs = args.runs

    if not os.path.exists(db_path):
        print(f"❌ Ошибка: БД не найдена: {db_path}")
        return
    
    print(f"📊 Подключение к базе: {db_path}")
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT val_loss FROM runs WHERE val_loss IS NOT NULL ORDER BY rowid ASC", conn)
        conn.close()
    except Exception as e:
        print(f"❌ Ошибка чтения БД: {e}")
        return

    if len(df) < 5:
        print("⚠️ Слишком мало данных.")
        return

    # ==========================================================
    # УМНОЕ АВТО-ОПРЕДЕЛЕНИЕ КОРИДОРА
    # ==========================================================
    if args.valid_max is None:
        q1 = df['val_loss'].quantile(0.25)
        q3 = df['val_loss'].quantile(0.75)
        iqr = q3 - q1
        
        # Классическая граница выбросов
        statistical_max = q3 + 1.5 * iqr
        
        # Страховка: отсекаем 15% самых жутких ранов в любом случае
        p85 = df['val_loss'].quantile(0.85)
        
        # Выбираем самую надежную (низкую) границу
        valid_max = min(statistical_max, p85)
        print(f"🧠 Авто-определение хаоса: верхняя граница (valid_max) установлена на {valid_max:.4f}")
    else:
        valid_max = args.valid_max

    valid_min = args.valid_min if args.valid_min is not None else 0.0
    # ==========================================================

    df['best_so_far'] = df['val_loss'].cummin()
    x_data = np.arange(1, len(df) + 1)
    y_data = df['best_so_far'].values

    valid_mask = (y_data >= valid_min) & (y_data <= valid_max)
    x_fit = x_data[valid_mask]
    y_fit = y_data[valid_mask]

    if len(x_fit) < 3:
        print(f"❌ Недостаточно данных в коридоре Loss от {valid_min} до {valid_max:.4f} для расчета тренда.")
        return

    initial_guess = [y_fit[0] - y_fit[-1], 0.05, y_fit[-1] - 0.01]
    lower_bounds = [0, 0, 0] 
    upper_bounds = [np.inf, np.inf, y_fit[-1]] 
    
    predicted_loss = None
    try:
        popt, pcov = curve_fit(
            decay_model, 
            x_fit, 
            y_fit, 
            p0=initial_guess, 
            bounds=(lower_bounds, upper_bounds), 
            maxfev=10000
        )
        a, b, c = popt
        ignored_runs = x_fit[0] - 1 
        
        predicted_loss = decay_model(target_runs, a, b, c)
        
        print("-" * 50)
        print(f"📈 РЕЗУЛЬТАТЫ АНАЛИЗА (Авто-проигнорировано хаоса: {ignored_runs} ранов):")
        print(f"Теоретический предел Loss (Асимптота): {c:.4f}")
        print(f"🎯 ПРОГНОЗ: К {target_runs}-му рану ожидаемый лучший Loss составит ~{predicted_loss:.4f}")
        print("-" * 50)
    except RuntimeError:
        print("❌ Не удалось подобрать кривую даже на отфильтрованных данных.")
        return

    max_plot_x = args.max_x if args.max_x else max(len(df) * 1.5, target_runs * 1.1)

    plt.figure(figsize=(12, 7))
    
    chaotic_mask = df['val_loss'] > valid_max
    plt.scatter(x_data[~chaotic_mask], df['val_loss'][~chaotic_mask], alpha=0.3, color='gray', label='Адекватные раны (Шум)')
    plt.scatter(x_data[chaotic_mask], df['val_loss'][chaotic_mask], alpha=0.5, color='red', marker='x', label='Хаос (Игнор)')
    
    y_plot_best = np.clip(y_data, a_min=None, a_max=valid_max)
    plt.step(x_data, y_plot_best, where='post', color='blue', linewidth=2, label='Best So Far (в коридоре)')
    
    x_pred = np.linspace(x_fit[0], max_plot_x, 500)
    y_pred = decay_model(x_pred, a, b, c)
    plt.plot(x_pred, y_pred, color='red', linestyle='--', linewidth=2, label='Аппроксимация (Тренд)')
    
    plt.axhline(y=c, color='purple', linestyle=':', label=f'Асимптота (Предел: {c:.4f})')
    
    plt.axvline(x=target_runs, color='green', alpha=0.5, label=f'Запрос: {target_runs} ранов')
    plt.axhline(y=predicted_loss, color='green', linestyle='-.', label=f'Прогноз: {predicted_loss:.4f}')
    plt.scatter([target_runs], [predicted_loss], color='green', zorder=5, s=100) 

    plt.title(f"Прогноз сходимости Loss (Умный фильтр хаоса) | Фолд: {os.path.basename(os.path.normpath(args.fold_dir))}")
    plt.xlabel("Количество ранов (Эксперименты)")
    plt.ylabel("Validation Loss")
    
    if args.max_x:
        plt.xlim(0, args.max_x)

    if args.max_y:
        y_upper_bound = args.max_y
    else:
        adequate_losses = df['val_loss'][~chaotic_mask]
        if len(adequate_losses) > 5:
            q1_ad = adequate_losses.quantile(0.25)
            q3_ad = adequate_losses.quantile(0.75)
            iqr_ad = q3_ad - q1_ad
            smart_max = q3_ad + 1.5 * iqr_ad
            
            y_upper_bound = max(smart_max, y_fit[0])
            if predicted_loss:
                y_upper_bound = max(y_upper_bound, predicted_loss)
            y_upper_bound += 0.02 
        else:
            y_upper_bound = valid_max

    y_lower_bound = min(y_fit)
    if 'c' in locals():
        y_lower_bound = min(y_lower_bound, c)
    if predicted_loss is not None:
        y_lower_bound = min(y_lower_bound, predicted_loss)
    
    y_lower_bound -= 0.02 
    
    plt.ylim(y_lower_bound, y_upper_bound)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()