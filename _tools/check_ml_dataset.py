import pandas as pd
from pathlib import Path
import sys
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent

def check_ml_dataset(exp_name):
    ML_DATA_DIR = BASE_DIR / "experiments" / exp_name / "dataset"
    print(f"🔍 Аудит датасета для {exp_name}...\n")
    
    train_path = ML_DATA_DIR / "train_scaled.csv"
    test_path = ML_DATA_DIR / "test_scaled.csv"
    if not train_path.exists() or not test_path.exists(): return
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"📅 ХРОНОЛОГИЯ")
    print(f"Train: {train_df['datetime'].min()} -> {train_df['datetime'].max()}")
    print(f"Test:  {test_df['datetime'].min()} -> {test_df['datetime'].max()}")
    
    print("\n⚖️ БАЛАНС КЛАССОВ")
    def print_bal(df, name):
        counts = df['label'].value_counts(normalize=True) * 100
        print(f"{name:5}: Флэт {counts.get(0.0,0):.1f}% | TP {counts.get(1.0,0):.1f}% | SL {counts.get(2.0,0):.1f}%")
    print_bal(train_df, "Train")
    print_bal(test_df, "Test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    args = parser.parse_args()
    check_ml_dataset(args.exp_name)