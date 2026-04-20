import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# --- Настройка путей ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "2_results"

def main(args):
    print(f"--- 🩺 Диагностика подготовленных данных для модели: {args.model_name} ---")
    
    data_path = RESULTS_DIR / "final_for_model" / args.model_name / "train_final.csv"
    
    if not data_path.exists():
        print(f"❌ Ошибка: Файл не найден по пути: {data_path}")
        print("Пожалуйста, убедитесь, что вы запустили скрипт assemble_final_data.py для этой модели.")
        return

    print(f"✅ Загружаю данные из: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"❌ Не удалось прочитать CSV файл: {e}")
        return

    # Определяем колонки с признаками (исключаем служебные)
    exclude_cols = {'datetime', 'ticker', 'label', 'meta_target'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not feature_cols:
        print("❌ Не найдено колонок с признаками в файле.")
        return

    print(f"Найдено {len(feature_cols)} признаков. Провожу анализ...")

    # --- Основная диагностика ---
    stats = df[feature_cols].describe().transpose()
    
    # Добавляем проверку на NaN и Inf
    stats['nan_count'] = df[feature_cols].isnull().sum()
    stats['inf_count'] = df[feature_cols].apply(lambda x: np.isinf(x).sum())
    
    # Ищем потенциальные проблемы
    # Выбросы: очень большое стандартное отклонение по сравнению со средним
    stats['is_outlier'] = (stats['std'] > 1000) & (stats['std'] > abs(stats['mean']) * 10)
    # Константы: стандартное отклонение равно нулю
    stats['is_constant'] = stats['std'] == 0
    
    problematic_features = stats[
        (stats['nan_count'] > 0) | 
        (stats['inf_count'] > 0) | 
        (stats['is_outlier']) |
        (stats['is_constant'])
    ]

    print("\n" + "="*80)
    if problematic_features.empty:
        print("🎉 Проверка завершена. Явных проблем (NaN, Inf, выбросы, константы) не найдено.")
        print("Статистика по всем признакам:")
        print(stats[['mean', 'std', 'min', 'max']].to_string())
    else:
        print("❗️ ВНИМАНИЕ! Обнаружены потенциально проблемные признаки:")
        print(problematic_features[['mean', 'std', 'min', 'max', 'nan_count', 'inf_count', 'is_outlier']].to_string())
        print("\nПолная статистика:")
        print(stats.to_string())
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Диагностика финального датасета на наличие аномалий.")
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        help="Имя модели, чьи данные нужно проверить (например, 'universal_60_10_v1_35_features')."
    )
    args = parser.parse_args()
    main(args)