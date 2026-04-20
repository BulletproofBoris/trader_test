# _tools/plot_landscape_from_data.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import os
import warnings
import pandas as pd

# --- Подавление лишних сообщений ---
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# --- Автономное определение путей ---
try:
    PROJECT_DIR = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_DIR = Path.cwd()

RESULTS_DIR = PROJECT_DIR / "2_results"
LANDSCAPES_DIR = RESULTS_DIR / "landscape_data"
# --- Конец блока ---

def main(args):
    print(f"--- ОТРИСОВКА ({args.plot_type}) ландшафта для модели '{args.model_name}' ---")
    
    landscape_data_dir = LANDSCAPES_DIR / args.model_name
    reports_dir = RESULTS_DIR / "reports" / f"landscape_plots_{args.model_name}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Ищем файлы, соответствующие указанному model_filename
    model_basename = Path(args.model_filename).stem
    data_files = sorted(list(landscape_data_dir.glob(f"landscape_data_{model_basename}_run_*.npz")))
    
    if not data_files:
        print(f"❌ В папке {landscape_data_dir} не найдены файлы .npz для '{args.model_filename}'.")
        print("   Убедитесь, что вы сначала запустили 'calculate_landscape.py' с этим же --model_filename.")
        return
        
    print(f"Найдено {len(data_files)} файлов с данными. Начинаю отрисовку...")

    for i, file_path in enumerate(data_files):
        run_id = i + 1
        data = np.load(file_path, allow_pickle=True)
        
        # Загружаем данные по правильным ключам и восстанавливаем 2D-сетку
        x_flat = data['x']
        y_flat = data['y']
        loss_flat = data['loss']
        
        points_per_axis = int(np.sqrt(len(loss_flat)))
        x_coords = np.unique(x_flat)
        y_coords = np.unique(y_flat)
        loss_grid = loss_flat.reshape(points_per_axis, points_per_axis)

        if args.plot_type == '2d':
            fig, ax = plt.subplots(figsize=(12, 10))
            
            min_loss = loss_grid.min()
            vmax_final = args.vmax
            if min_loss >= vmax_final:
                print(f"⚠️  Предупреждение (срез #{run_id}): Min loss ({min_loss:.4f}) >= Vmax ({vmax_final:.4f}). Vmax будет увеличен.")
                vmax_final = loss_grid.max()

            levels = np.linspace(min_loss, vmax_final, 25)
            contour = ax.contourf(x_coords, y_coords, loss_grid, levels=levels, cmap=cm.viridis, extend='max')
            
            fig.colorbar(contour, ax=ax, label='val_loss')
            ax.plot([0], [0], 'rX', markersize=12, label='Центр (обученная модель)')
            ax.set_title(f"Ландшафт потерь (2D) - {model_basename} - Срез #{run_id}", fontsize=16)
            ax.set_xlabel("Направление 1")
            ax.set_ylabel("Направление 2")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            filename = f"landscape_plot_{model_basename}_2D_run_{run_id}.png"

        elif args.plot_type == '3d':
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(x_coords, y_coords)
            ax.plot_surface(X, Y, loss_grid.T, cmap=cm.viridis, alpha=0.8)
            
            center_loss_idx = len(x_coords) // 2
            center_loss = loss_grid[center_loss_idx, center_loss_idx]
            ax.plot([0], [0], [center_loss], 'rX', markersize=12, label='Центр (обученная модель)')
            
            ax.set_title(f"Ландшафт потерь (3D) - {model_basename} - Срез #{run_id}", fontsize=16)
            ax.set_xlabel("Направление 1")
            ax.set_ylabel("Направление 2")
            ax.set_zlabel("val_loss")
            ax.legend()

            filename = f"landscape_plot_{model_basename}_3D_run_{run_id}.png"
        
        output_path = reports_dir / filename
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        
    print(f"\n✅ Отрисовка завершена. {len(data_files)} изображений сохранены в: {reports_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Отрисовка ландшафта по рассчитанным данным.")
    parser.add_argument('--model_name', type=str, required=True, help="Общее имя модели (папка).")
    parser.add_argument('--model_filename', type=str, default="model.keras", help="Имя файла модели, для которой были рассчитаны данные.")
    parser.add_argument('--plot_type', type=str, required=True, choices=['2d', '3d'], help="Тип графика для отрисовки.")
    parser.add_argument('--vmax', type=float, default=1.0, help="Максимальное значение val_loss для цветовой шкалы (только для 2D).")
    
    args = parser.parse_args()
    main(args)