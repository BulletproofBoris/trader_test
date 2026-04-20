import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = Path(__file__).resolve().parent.parent

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def process_and_write(df, lookback, output_path, feature_cols):
    writer = tf.io.TFRecordWriter(str(output_path))
    samples_count = 0
    all_labels = []
    
    for ticker, group in tqdm(df.groupby('ticker'), desc=f"Запись {output_path.name}"):
        group = group.sort_values('datetime')
        features = group[feature_cols].values
        labels = group['label'].values
        
        for i in range(len(features) - lookback):
            seq = features[i : i + lookback]
            target = int(labels[i + lookback - 1]) 
            seq_bytes = tf.io.serialize_tensor(tf.cast(seq, tf.float32)).numpy()
            
            feature = {
                'sequence': _bytes_feature(seq_bytes),
                'target': _int64_feature(target)
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            
            all_labels.append(target)
            samples_count += 1
            
    writer.close()
    return samples_count, all_labels

def main(args):
    EXP_DIR = BASE_DIR / "experiments" / args.exp_name
    DATASET_DIR = EXP_DIR / "dataset"
    TFRECORDS_DIR = EXP_DIR / "tfrecords"
    TFRECORDS_DIR.mkdir(parents=True, exist_ok=True)
    
    config_path = EXP_DIR / "exp_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        exp_config = json.load(f)
    lookback = exp_config["lookback_bars"]
    
    features_json = DATASET_DIR / "features_selected.json"
    if not features_json.exists():
        print(f"❌ Файл features_selected.json не найден!"); return
    with open(features_json, 'r') as f:
        feature_cols = json.load(f)["feature_order"]

    print(f"🚀 Упаковка TFRecords для: {args.exp_name} (Окно: {lookback})")
    train_df = pd.read_csv(DATASET_DIR / "train_scaled.csv")
    test_df = pd.read_csv(DATASET_DIR / "test_scaled.csv")
    
    train_samples, train_labels = process_and_write(train_df, lookback, TFRECORDS_DIR / "train.tfrecord", feature_cols)
    test_samples, _ = process_and_write(test_df, lookback, TFRECORDS_DIR / "test.tfrecord", feature_cols)
    
    classes = np.unique(train_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weights_dict = {str(c): float(w) for c, w in zip(classes, weights)}
    
    meta_info = {
        "lookback": lookback,
        "n_features": len(feature_cols),
        "train_samples": train_samples,
        "test_samples": test_samples,
        "class_weights": class_weights_dict
    }
    with open(TFRECORDS_DIR / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=4)
        
    print(f"✅ Успешно! Train: {train_samples}, Test: {test_samples}. Веса: {class_weights_dict}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    args = parser.parse_args()
    main(args)