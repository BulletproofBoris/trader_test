import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import os
import warnings

# Отключаем системный спам scipy при промежуточных вычислениях (в бутстрэпе их будет много)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# МОДЕЛЬ АППРОКСИМАЦИИ (3 параметра: амплитуда, скорость, асимптота)
# ---------------------------------------------------------
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def main():
    parser = argparse.ArgumentParser(description="Анализатор макро-тренда (Bootstrap / Веер Уверенности).")
    parser.add_argument("fold_dir", type=str, help="Путь к папке фолда")
    parser.add_argument("--runs", type=int, default=100, help="Выделенный бюджет пула")
    parser.add_argument("--max_x", type=int, default=None, help="Ограничение по оси X")
    parser.add_argument("--valid_min", type=float, default=0.0, help="Нижняя граница")
    parser.add_argument("--valid_max", type=float, default=None, help="Верхняя граница")
    
    args = parser.parse_args()
    db_path = os.path.join(args.fold_dir, "trading_factory.db")
    total_budget = args.runs

    if not os.path.exists(db_path):
        print(f"❌ БД не найдена: {db_path}"); return
    
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT val_loss FROM runs WHERE status='COMPLETED' AND val_loss IS NOT NULL ORDER BY rowid ASC", conn)
        conn.close()
    except Exception as e:
        print(f"❌ Ошибка БД: {e}"); return

    if len(df) < 5:
        print("⚠️ Слишком мало данных."); return

    # ==========================================================
    # 1. АВТО-ФИЛЬТРАЦИЯ И ПОДГОТОВКА ДАННЫХ
    # ==========================================================
    if args.valid_max is None:
        q1, q3 = df['val_loss'].quantile(0.25), df['val_loss'].quantile(0.75)
        valid_max = min(q3 + 1.5 * (q3 - q1), df['val_loss'].quantile(0.85))
    else:
        valid_max = args.valid_max

    x_data = np.arange(1, len(df) + 1)
    y_data_raw = df['val_loss'].values
    
    # Отсеиваем жестокий хаос
    valid_mask = (y_data_raw <= valid_max) & (y_data_raw >= args.valid_min)
    x_fit = x_data[valid_mask]
    y_fit_raw = y_data_raw[valid_mask]

    if len(x_fit) < 5:
        print("❌ Недостаточно данных после фильтрации."); return

    # Вычисляем реально наблюдаемую амплитуду (размах)
    obs_amp = max(1e-4, np.max(y_fit_raw) - np.min(y_fit_raw))
    amplitude_guess = obs_amp
    
    # Готовим график
    max_x = args.max_x if args.max_x else max(len(df)*1.5, total_budget)
    x_range = np.linspace(1, max_x, 500)
    plt.figure(figsize=(16, 9))

    # Рисуем сырые точки
    chaotic_mask = y_data_raw > valid_max
    plt.scatter(x_data[~chaotic_mask], y_data_raw[~chaotic_mask], alpha=0.4, color='gray', label='Адекватные раны')
    plt.scatter(x_data[chaotic_mask], y_data_raw[chaotic_mask], alpha=0.3, color='red', marker='x', label='Хаос (игнор)')
    plt.step(x_data, df['val_loss'].cummin(), where='post', color='black', alpha=0.4, linewidth=2, label='Фактический Рекорд (cummin)')

    # ==========================================================
    # 2. ДЕТЕКТОР ЛЖИ: BOOTSTRAPPING (100 реальностей)
    # ==========================================================
    bootstrap_a, bootstrap_b, bootstrap_c = [], [], []
    
    # СТРОГИЙ ДИНАМИЧЕСКИЙ ПОВОДОК:
    # Вычисляем реально наблюдаемую амплитуду (размах)
    obs_amp = max(1e-4, np.max(y_fit_raw) - np.min(y_fit_raw))
    amplitude_guess = obs_amp
    
    # 1. Асимптота не может улететь вниз больше, чем на 2 размаха текущего прогресса.
    # Если прогресс был 0.01, мы разрешаем заглянуть вниз максимум на 0.02.
    min_c = max(0.0, np.min(y_fit_raw) - (obs_amp * 2.0))
    
    # 2. Амплитуда 'a' не должна превышать 3-кратный наблюдаемый размах.
    max_a = max(1e-3, obs_amp * 3.0)

    # 3. Скорость 'b' СТРОГО больше 0.01 (запрещаем оптимизатору рисовать плоские линии)
    free_bounds = (
        [1e-5, 0.01, min_c], 
        [max_a, 2.0, np.max(y_fit_raw)]
    )
    
    successful_boots = 0
    N_BOOTSTRAPS = 100
    SAMPLE_FRACTION = 0.8 # Берем 80% данных для каждой реальности

    for _ in range(N_BOOTSTRAPS):
        # Случайно выбираем 80% индексов, сохраняя ход времени (np.sort обязательно!)
        sample_size = max(5, int(len(x_fit) * SAMPLE_FRACTION))
        sample_indices = np.sort(np.random.choice(len(x_fit), size=sample_size, replace=False))
        
        x_boot = x_fit[sample_indices]
        y_raw_boot = y_fit_raw[sample_indices]
        
        # Строим НОВУЮ лестницу рекордов для этой альтернативной реальности
        y_cummin_boot = np.minimum.accumulate(y_raw_boot)
        
        p0 = [amplitude_guess, 0.05, max(0, np.min(y_cummin_boot) - 0.01)]
        
        try:
            popt, _ = curve_fit(exp_func, x_boot, y_cummin_boot, p0=p0, bounds=free_bounds, maxfev=2000)
            a_b, b_b, c_b = popt
            
            bootstrap_a.append(a_b)
            bootstrap_b.append(b_b)
            bootstrap_c.append(c_b)
            
            # Рисуем нить веера (очень прозрачную)
            y_pred = exp_func(x_range, a_b, b_b, c_b)
            plt.plot(x_range, y_pred, color='red', alpha=0.04) 
            successful_boots += 1
        except:
            pass

    # ==========================================================
    # 3. АНАЛИЗ ВЕЕРА И ВЫЧИСЛЕНИЕ ИСТИНЫ
    # ==========================================================
    print(f"\n📊 АНАЛИЗ ФОЛДА: {os.path.basename(args.fold_dir)}")
    print(f"🥇 Текущий абсолютный рекорд (Loss): {np.min(y_fit_raw):.4f}")
    
    if successful_boots < 10:
        print("❌ Бутстрэппинг провалился (недостаточно стабильных данных).")
    else:
        # Истинная асимптота — это медиана из 100 реальностей
        true_c = np.median(bootstrap_c)
        true_a = np.median(bootstrap_a)
        true_b = np.median(bootstrap_b)
        
        # Считаем разброс (уверенность алгоритма)
        q5_c = np.percentile(bootstrap_c, 5)
        q95_c = np.percentile(bootstrap_c, 95)
        uncertainty = q95_c - q5_c
        
        print(f"\n🎯 Истинная макро-асимптота (Медиана): {true_c:.4f}")
        print(f"📊 Доверительный интервал (90%): от {q5_c:.4f} до {q95_c:.4f}")
        print(f"⚖️ Ширина неопределенности: {uncertainty:.5f}")
        
        if uncertainty < 0.005:
            print("\n✅ ВЕРДИКТ: Оценка ПРАВДИВА. Веер сошелся, тренд железобетонный.")
            # Если мы пробили нижнюю границу доверительного интервала, значит точно дно.
            if np.min(y_fit_raw) <= q5_c:
                 print("🛑 ОРКЕСТРАТОР: Целесообразно закрыть фолд (Рекорд пробил 5% квантиль асимптоты).")
            else:
                 print("🟢 ОРКЕСТРАТОР: Можно продолжать (Рекорд еще не достиг медианной асимптоты).")
        else:
            print("\n⚠️ ВЕРДИКТ: Оценка СОМНИТЕЛЬНА. Широкий веер, математика не уверена в пределе. Нужны еще раны.")

        # Рисуем главную (Истинную) линию жирным
        y_true_pred = exp_func(x_range, true_a, true_b, true_c)
        plt.plot(x_range, y_true_pred, color='darkred', linewidth=3, label=f'Истинный Тренд (Медиана)')
        
        # Рисуем зону неопределенности асимптоты
        plt.axhline(y=true_c, color='darkred', linestyle='--', linewidth=2, label=f'Истинная Асимптота ({true_c:.4f})')
        plt.fill_between(x_range, q5_c, q95_c, color='red', alpha=0.15, label=f'Доверительный коридор асимптоты (90%)')

    plt.axvline(x=total_budget, color='blue', linestyle='-.', alpha=0.4, label='Бюджет пула')
    
    plt.title("Детектор Лжи HPO (Bootstrapping / 100 Реальностей)")
    plt.xlabel("Раны (Эксперименты)"); plt.ylabel("Validation Loss")
    
    # Хак для красивой легенды, чтобы не дублировать 100 нитей
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.grid(alpha=0.3, linestyle='--')
    
    y_lim_bottom = q5_c - 0.01 if successful_boots >= 10 else np.min(y_fit_raw) - 0.02
    plt.ylim(y_lim_bottom, valid_max)
    plt.show()

if __name__ == "__main__":
    main()