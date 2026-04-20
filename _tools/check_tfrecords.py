import os
import sys
import json
import argparse
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent

def parse_tfrecord_fn(example, lookback, n_features):
    feature_description = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed = tf.io.parse_single_example(example, feature_description)
    sequence = tf.io.parse_tensor(parsed['sequence'], out_type=tf.float32)
    sequence.set_shape([lookback, n_features])
    return sequence, parsed['target']

def check_tfrecords(exp_name):
    TFRECORDS_DIR = BASE_DIR / "experiments" / exp_name / "tfrecords"
    print(f"🔍 Аудит бинарных файлов TFRecord для {exp_name}...\n")
    
    meta_path = TFRECORDS_DIR / "metadata.json"
    if not meta_path.exists(): return
    with open(meta_path, 'r') as f: meta = json.load(f)
        
    lookback, n_features = meta['lookback'], meta['n_features']
    
    for ds_type in ['train', 'test']:
        filepath = TFRECORDS_DIR / f"{ds_type}.tfrecord"
        if not filepath.exists(): continue
            
        print(f"🛠️ ПРОВЕРКА {filepath.name}:")
        raw_dataset = tf.data.TFRecordDataset(str(filepath))
        dataset = raw_dataset.map(lambda x: parse_tfrecord_fn(x, lookback, n_features))
        
        for seq, target in dataset.take(1):
            print(f"  ✅ Форма X: {tuple(seq.shape)}")
            print(f"  ✅ Форма Y: {tuple(target.shape)} -> Класс {target.numpy()}")
            
        count = sum(1 for _ in raw_dataset)
        expected = meta[f'{ds_type}_samples']
        if count == expected: print(f"  ✅ Сэмплы сходятся: {count:,} шт.\n")
        else: print(f"  ❌ Ошибка: ожидалось {expected}, найдено {count}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    args = parser.parse_args()
    check_tfrecords(args.exp_name)