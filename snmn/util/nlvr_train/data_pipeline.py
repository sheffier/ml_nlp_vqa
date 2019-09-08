import tensorflow as tf
import multiprocessing


#===============DEFINE YOUR ARGUMENTS==============
class Flags:
  def __init__(self):
    pass


FLAGS = Flags()
FLAGS.max_seq_length = 54
FLAGS.max_predictions_per_seq = 8


def create_dataset(file_pattern, batch_size, max_seq_length, max_predictions_per_seq,
                   is_training, num_epochs=1):
    def _parse(record):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
            "seq_length": tf.FixedLenFeature([], tf.int64),
            "masked_lm_positions": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights": tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "answer": tf.FixedLenFeature([], tf.int64),
            "image_id": tf.FixedLenFeature([], tf.int64),
            "img_features": tf.FixedLenFeature([7, 14, 2048], tf.float32)}

        example = tf.parse_single_example(record, name_to_features)

        return {
            "input_ids": tf.cast(example["input_ids"], tf.int32),
            "seq_length": tf.cast(example["seq_length"], tf.int32),
            "masked_lm_positions": tf.cast(example["masked_lm_positions"], tf.int32),
            "masked_lm_ids": tf.cast(example["masked_lm_ids"], tf.int32),
            "masked_lm_weights": example["masked_lm_weights"],
            "answer": tf.cast(example["answer"], tf.int32),
            "image_id": tf.cast(example["image_id"], tf.int32),
            "img_features": example["img_features"]}

    with tf.device('/cpu:0'):
        # generate file list
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)

        # `cycle_length` is the number of parallel files that get read.
        threads = multiprocessing.cpu_count()
        cycle_length = min(threads, len(file_pattern))

        dataset = dataset.interleave(tf.data.TFRecordDataset,
                                     cycle_length=cycle_length,
                                     block_length=1,
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.repeat(num_epochs)

        if is_training is True:
            dataset = dataset.shuffle((20*batch_size))

        dataset = dataset.map(_parse, num_parallel_calls=threads)

        dataset = dataset.batch(batch_size=batch_size, drop_remainder=is_training)

        dataset = dataset.prefetch(buffer_size=batch_size)

    return dataset


def prepare_dataset_iterators(train_file_pattern, val_file_pattern, batch_size=128):
    # Make a dataset from the train data
    train_ds = create_dataset(train_file_pattern, batch_size, FLAGS.max_seq_length, FLAGS.max_predictions_per_seq,
                              True)
    # Make a dataset from the train data
    val_ds = create_dataset(val_file_pattern, batch_size, FLAGS.max_seq_length, FLAGS.max_predictions_per_seq,
                            False)

    # Define an abstract iterator
    # Make an iterator object that has the shape and type of our datasets
    iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                               train_ds.output_shapes)

    # This is an op that gets the next element from the iterator
    next_element = iterator.get_next()
    # These ops let us switch and reinitialize every time we finish an epoch
    training_init_op = iterator.make_initializer(train_ds)
    validation_init_op = iterator.make_initializer(val_ds)

    return next_element, training_init_op, validation_init_op
