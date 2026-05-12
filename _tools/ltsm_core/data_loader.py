import tensorflow as tf

def parse_tfrecord_fn(example, lookback, n_features):
    feature_description = {
        'sequence': tf.io.FixedLenFeature([], tf.string), 
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    sequence = tf.io.parse_tensor(example['sequence'], out_type=tf.float32)
    sequence.set_shape([lookback, n_features])
    label = tf.one_hot(example['target'], depth=3)
    label.set_shape([3])
    
    return sequence, label

def load_tfrecord_dataset(file_path, batch_size, lookback, n_features, is_training=True):
    dataset = tf.data.TFRecordDataset(str(file_path), num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: parse_tfrecord_fn(x, lookback, n_features), num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        # Кешируем данные в оперативную память (если её хватает!)
        # Если ОЗУ мало (меньше 32 ГБ), эту строчку лучше закомментировать
        dataset = dataset.cache() 
        dataset = dataset.shuffle(buffer_size=8192, reshuffle_each_iteration=True)
    else:
        dataset = dataset.cache()
        
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    
    # Заставляем процессор готовить следующий батч, пока GPU считает текущий
    dataset = dataset.prefetch(tf.data.AUTOTUNE) 
    
    return dataset

def compute_class_weights_fast(tfrecord_path):
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    class_counts = {0: 0, 1: 0, 2: 0}
    feature_description = {'target': tf.io.FixedLenFeature([], tf.int64)}
    
    for raw_record in dataset:
        parsed = tf.io.parse_single_example(raw_record, feature_description)
        class_counts[int(parsed['target'].numpy())] += 1
        
    total = sum(class_counts.values())
    weights = {c: total / (3.0 * max(1, count)) for c, count in class_counts.items()}
    return weights

def count_tfrecord_samples(tfrecord_path):
    """Считает количество строк в датасете для расчета батча"""
    return sum(1 for _ in tf.data.TFRecordDataset(str(tfrecord_path)))