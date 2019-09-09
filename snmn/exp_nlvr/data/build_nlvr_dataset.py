import json
import math
import os
import random
import re
import sys
import collections
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from multiprocessing import Pool

sys.path.append('../../')  # NOQA
from util import text_processing

examples_file = '../nlvr_dataset/{set}.json'

image_dir = '../nlvr_images/images/%s/'
feature_dir = './resnet152_c5_7x7/%s/'
datasets_dir = './tfrecords'
question_vocab_file = './vocabulary_nlvr.txt'
answer_vocab_file = './answers_nlvr.txt'
global_max_len = 0


class TrainingInstance(object):
    def __init__(self, tokens, answer, masked_lm_positions, masked_lm_labels,
                 image_id, feature_path):
        self.tokens = tokens
        self.answer = answer
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.image_id = image_id
        self.feature_path = feature_path


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_from_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def _float_feature_from_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))


def get_tfrecord_filename(output_path, split_name, shard_id, num_shards):
    output_filename = '%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, num_shards)
    return os.path.join(output_path, output_filename)


def convert_to_tfrecord(instances, output_filename, max_seq_length, max_predictions_per_seq,
                        qst_vocab_dict, ans_vocab_dict):
    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(output_filename) as writer:
        for instance in instances:
            image_id = instance.image_id
            img_features = np.load(instance.feature_path, allow_pickle=True)
            img_features = np.squeeze(img_features)
            # height, width, channels = img_features.shape
            img_features = img_features.flatten()
            # img_features_bytes = img_features.to_string()
            answer = ans_vocab_dict.word2idx(instance.answer)
            input_ids = np.array(
                [qst_vocab_dict.word2idx(w) for w in instance.tokens])

            seq_length = len(input_ids)

            assert seq_length <= max_seq_length
            input_ids = np.concatenate((input_ids, np.zeros(max_seq_length-seq_length, dtype=input_ids.dtype)))
            assert len(input_ids) == max_seq_length

            num_preds = len(instance.masked_lm_labels)
            masked_lm_positions = np.copy(instance.masked_lm_positions)
            masked_lm_ids = np.array([qst_vocab_dict.word2idx(w) for w in instance.masked_lm_labels])
            masked_lm_weights = np.ones(num_preds, dtype=np.float)

            zeros = np.zeros(max_predictions_per_seq - num_preds)
            masked_lm_positions = np.concatenate((masked_lm_positions, zeros.astype(masked_lm_positions.dtype)))
            masked_lm_ids = np.concatenate((masked_lm_ids, zeros.astype(masked_lm_ids.dtype)))
            masked_lm_weights = np.concatenate((masked_lm_weights, zeros.astype(masked_lm_weights.dtype)))
            assert len(masked_lm_positions) == max_predictions_per_seq, (len(masked_lm_positions), max_predictions_per_seq)
            assert len(masked_lm_ids) == max_predictions_per_seq, (len(masked_lm_ids), max_predictions_per_seq)
            assert len(masked_lm_weights) == max_predictions_per_seq, (len(masked_lm_weights), max_predictions_per_seq)

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.

            features = collections.OrderedDict()
            features["input_ids"] = _int64_feature_from_list(input_ids)
            features["seq_length"] = _int64_feature(seq_length)
            features["masked_lm_positions"] = _int64_feature_from_list(masked_lm_positions)
            features["masked_lm_ids"] = _int64_feature_from_list(masked_lm_ids)
            features["masked_lm_weights"] = _float_feature_from_list(masked_lm_weights)
            features["answer"] = _int64_feature(answer)
            features["image_id"] = _int64_feature(image_id)
            features["img_features"] = _float_feature_from_list(img_features)

            # Wrap as a TensorFlow SequenceExample.
            example = tf.train.Example(features=tf.train.Features(feature=features))

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq,
                                 qst_vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    output_tokens = list(tokens)

    if rng is not None:
        cand_indexes = []
        for (i, token) in enumerate(tokens):
            cand_indexes.append(i)

        rng.shuffle(cand_indexes)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(tokens) * masked_lm_prob))))

        masked_lms = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "<mask>"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = qst_vocab_words[rng.randint(0, len(qst_vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p.index)
            masked_lm_labels.append(p.label)
    else:
        masked_lm_positions = np.empty(max_predictions_per_seq, dtype=np.int32)
        masked_lm_labels = np.empty(max_predictions_per_seq, dtype=np.int32)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def create_dataset_instances(base_imdb, split_name, dupe_factor, masked_lm_prob,
                             max_predictions_per_seq, qst_vocab_words, rng):
    instances = []

    with tqdm(total=dupe_factor * len(base_imdb)) as pbar:
        pbar.set_description("[%s] creating instance dataset" % split_name)
        for _ in range(dupe_factor):
            for iminfo in base_imdb:
                tokens = iminfo["question_tokens"]
                answer = iminfo['answer']
                image_id = iminfo["image_id"]
                feature_path = iminfo["feature_path"]

                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, qst_vocab_words, rng)

                instance = TrainingInstance(
                    tokens=tokens,
                    answer=answer,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                    image_id=image_id,
                    feature_path=feature_path
                )
                instances.append(instance)

                pbar.update(1)

    return instances


def build_base_imdb(image_set):
    global global_max_len

    print('building imdb %s' % image_set)
    load_answer = True
    assert image_set in ['train', 'dev', 'test1']
    with open(examples_file.format(set=image_set)) as f:
        examples = [json.loads(line) for line in f if line]

    imdb = []

    image_pair_regex = re.compile(r'^[-\w]+(?=-\d)')
    max_len = 0

    for image_id, example in enumerate(tqdm(examples, file=sys.stdout)):
        question_id = example['identifier']

        image_name = image_pair_regex.match(question_id).group(0)
        images_dir = os.path.abspath(os.path.join(image_dir % image_set))
        image_path = os.path.join(images_dir, f'{image_name}.png')
        feature_path = os.path.abspath(os.path.join(feature_dir % image_set, f'{image_name}.npy'))

        missing_images = not os.path.isfile(image_path)
        if missing_images:
            tqdm.write("Image Missing:\t" + image_path)
            continue

        if not os.path.isfile(feature_path):
            tqdm.write("Image Features Missing:\t" + feature_path)
            continue

        question_str = example['sentence']
        question_tokens = text_processing.tokenize(question_str)

        if len(question_tokens) > max_len:
            max_len = len(question_tokens)

        iminfo = dict(image_name=image_name,
                      image_path=image_path,
                      image_id=image_id,
                      question_id=question_id,
                      feature_path=feature_path,
                      question_str=question_str,
                      question_tokens=question_tokens)

        # load answers
        if load_answer:
            assert example['label'] in ("True", "False")  # Assumption: answer is always "True"/"False"
            iminfo['answer'] = example['label']

        imdb.append(iminfo)

    print("[%s] Max seq length: %d" % (image_set, max_len))

    if max_len > global_max_len:
        global_max_len = max_len

    return np.array(imdb)


def convert_instances_to_tfrecords(split_name, instances, num_instances_per_shard, max_seq_length,
                                   max_predictions_per_seq, qst_vocab_dict, ans_vocab_dict):
    n_instances = len(instances)
    num_shards = int(math.ceil(n_instances / num_instances_per_shard))

    pool = Pool()
    pbar = tqdm(total=num_shards)

    pbar.set_description("Converting %s imdb to tfrecords: " % split_name)

    def update(*a):
        pbar.update()

    for i in range(pbar.total):
        output_filename = get_tfrecord_filename(datasets_dir, split_name, i, num_shards)
        inst_slice = instances[i * num_instances_per_shard:i * num_instances_per_shard + num_instances_per_shard]
        pool.apply_async(convert_to_tfrecord,
                         args=(inst_slice, output_filename, max_seq_length, max_predictions_per_seq,
                               qst_vocab_dict, ans_vocab_dict),
                         callback=update)
    pool.close()
    pool.join()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='training',
                        help='Type of dataset to generate. Type = "training"/"pretraining"')
    parser.add_argument('--seed', default=0)

    args = parser.parse_args()

    assert args.type in ("training", "pretraining"),\
        "Wrong dataset type. Expected ('training', 'pretraining'), got %s" % args.type

    is_pretrain = (args.type == 'pretraining')
    random_seed = args.seed

    qst_vocab_dict = text_processing.VocabDict(question_vocab_file)
    ans_vocab_dict = text_processing.VocabDict(answer_vocab_file)
    masked_lm_prob = 0.15
    max_predictions_per_seq = 8

    if is_pretrain:
        dupe_factor = 5
        rng = random.Random(random_seed)
        name_prefix = 'masked_'
    else:
        dupe_factor = 1
        rng = None
        name_prefix = ''

    imdb_train = build_base_imdb('train')
    imdb_dev = build_base_imdb('dev')
    imdb_test = build_base_imdb('test1')

    os.makedirs('./imdb_r152_7x7', exist_ok=True)
    np.save('./imdb_r152_7x7/imdb_train.npy', imdb_train)
    np.save('./imdb_r152_7x7/imdb_dev.npy', imdb_dev)
    np.save('./imdb_r152_7x7/imdb_test.npy', imdb_test)

    train_instances = create_dataset_instances(imdb_train, name_prefix + 'train', dupe_factor, masked_lm_prob,
                                               max_predictions_per_seq, qst_vocab_dict.word_list, rng)

    if rng is not None:
        rng.shuffle(train_instances)

    dev_instances = create_dataset_instances(imdb_dev,  name_prefix + 'dev', dupe_factor, masked_lm_prob,
                                             max_predictions_per_seq, qst_vocab_dict.word_list, rng)
    # test_instances = create_dataset_instances(imdb_test, 'test', masked_lm_prob, max_predictions_per_seq,
    #                                           question_vocab_dict.word_list, rng)

    os.makedirs(datasets_dir, exist_ok=True)

    num_instances_per_shard = 150

    convert_instances_to_tfrecords(name_prefix + 'train', train_instances, num_instances_per_shard, global_max_len,
                                   max_predictions_per_seq, qst_vocab_dict, ans_vocab_dict)
    convert_instances_to_tfrecords(name_prefix + 'dev', dev_instances, num_instances_per_shard, global_max_len,
                                   max_predictions_per_seq, qst_vocab_dict, ans_vocab_dict)
    # convert_instances_to_tfrecords('test', test_instances, num_instances_per_shard, global_max_len,
    #                                max_predictions_per_seq, qst_vocab_dict, ans_vocab_dict)
