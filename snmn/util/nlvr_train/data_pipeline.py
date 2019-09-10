import multiprocessing
import tensorflow as tf
import numpy as np
from prefetch_generator import background


#===============DEFINE YOUR ARGUMENTS==============
class Flags:
  def __init__(self):
    pass


FLAGS = Flags()
FLAGS.max_seq_length = 54
FLAGS.max_predictions_per_seq = 8


class RandomMaskedDataset(object):
    def __init__(self, file_pattern, n_dups, n_shards_per_dup):
        file_list = sorted(tf.gfile.Glob(file_pattern))
        assert len(file_list) == n_dups * n_shards_per_dup

        self.n_dups = n_dups
        self.n_shards_per_dup = n_shards_per_dup
        self.file_array = np.array(file_list).reshape([n_dups, n_shards_per_dup])

        @background(max_prefetch=3)
        def dataset_gen():
            while True:
                rnd_idx = np.random.randint(0, self.n_dups, size=self.n_shards_per_dup)

                yield self.file_array[rnd_idx, np.arange(self.n_shards_per_dup)]

        self.gen = dataset_gen()

    def __next__(self):
        return self.gen.__next__()


def create_masked_dataset(file_list, batch_size, max_seq_length, max_predictions_per_seq,
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

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    # generate file list
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    dataset = dataset.with_options(option_no_order)

    dataset = dataset.shuffle(tf.size(file_list, out_type=tf.dtypes.int64))

    # `cycle_length` is the number of parallel files that get read.
    threads = multiprocessing.cpu_count()
    cycle_length = threads

    dataset = dataset.interleave(tf.data.TFRecordDataset,
                                 cycle_length=cycle_length,
                                 block_length=1,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.repeat(num_epochs)

    if is_training is True:
        dataset = dataset.shuffle((20*batch_size))

    dataset = dataset.map(_parse, num_parallel_calls=threads)

    # dataset = dataset.batch(batch_size=batch_size, drop_remainder=is_training)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


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

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    # generate file list
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)
    dataset = dataset.with_options(option_no_order)

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

    dataset = dataset.map(_parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def prepare_mask_dataset_iterators(train_file_list, val_file_list, batch_size=128):
    # Make a dataset from the train data
    train_ds = create_masked_dataset(train_file_list, batch_size, FLAGS.max_seq_length, FLAGS.max_predictions_per_seq,
                                     True)
    # Make a dataset from the train data
    val_ds = create_masked_dataset(val_file_list, batch_size, FLAGS.max_seq_length, FLAGS.max_predictions_per_seq,
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
    # validation_init_op = None

    return next_element, training_init_op, validation_init_op


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
