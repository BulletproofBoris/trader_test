# _tools/convert_to_tfrecord.py (Упрощенная версия)
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "2_results"

def _bytes_feature(value: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value: int):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def write_sequences_to_tfrecord(df: pd.DataFrame, feature_cols: list, lookback: int, output_path: Path):
    num_sequences = 0
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for ticker, group in tqdm(df.groupby('ticker'), desc=f"Writing {output_path.name}"):
            if len(group) < lookback: continue

            features = group[feature_cols].values.astype(np.float32)
            labels = group['label'].values.astype(np.int64)

            for i in range(len(group) - lookback + 1):
                sequence = features[i:i+lookback]
                target = labels[i + lookback - 1]
                
                seq_tensor = tf.convert_to_tensor(sequence)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'sequence': _bytes_feature(tf.io.serialize_tensor(seq_tensor).numpy()),
                    'target': _int64_feature(target)
                }))
                writer.write(example.SerializeToString())
                num_sequences += 1
    return num_sequences

def main(args):
    print(f"--- Converting scaled data to TFRecord for model: {args.model_name} ---")
    
    # --- Входные пути (теперь из папки features_scaled) ---
    source_dir = RESULTS_DIR / "features_scaled" / args.model_name
    train_path = source_dir / "train_scaled.csv"
    test_path = source_dir / "test_scaled.csv"

    # --- Выходные пути ---
    output_dir = RESULTS_DIR / f"tfrecord_{args.model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Загрузка списка признаков (теперь он лежит рядом со скейлером) ---
    features_path = RESULTS_DIR / "scalers" / f"{args.model_name}_features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}. Run fit_scaler first.")
        
    with open(features_path, 'r') as f:
        feature_cols = json.load(f)['feature_order']

    # --- Загрузка отмасштабированных данных ---
    print("Loading scaled data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # --- Запись TFRecord ---
    train_samples = write_sequences_to_tfrecord(train_df, feature_cols, args.lookback, output_dir / "train.tfrecord")
    test_samples = write_sequences_to_tfrecord(test_df, feature_cols, args.lookback, output_dir / "test.tfrecord")

    # --- Расчет весов и сохранение метаданных ---
    print("Calculating class weights and saving metadata...")
    class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
    class_weights_dict = {str(i): w for i, w in enumerate(class_weights)}

    metadata = {
        "model_name": args.model_name, "n_features": len(feature_cols), "lookback": args.lookback,
        "class_weights": class_weights_dict, "train_samples": train_samples,
        "val_samples": test_samples, "num_test_examples": test_samples
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ TFRecord files and metadata saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert scaled CSV data to TFRecord format.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lookback", type=int, default=60)
    args = parser.parse_args()
    main(args)