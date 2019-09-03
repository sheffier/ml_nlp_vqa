import json
import os
import re
import sys

import numpy as np
from tqdm import tqdm

sys.path.append('../../')  # NOQA
from util import text_processing
# from models_nlvr.config import cfg

question_vocab_file = './vocabulary_nlvr.txt'
examples_file = '../nlvr_dataset/{set}.json'

image_dir = '../nlvr_images/images/%s/'
feature_dir = './resnet152_c5_7x7/%s/'

max_len = 0


def pad_imdb(imdb, max_len, name):
    print('padding imdb %s to %d' % (name, max_len))
    for i in tqdm(range(len(imdb))):
        question_tokens = imdb[i]['question_tokens']
        pad_array = np.array([0] * (max_len - question_tokens.size))
        imdb[i]['question_tokens'] = np.concatenate((question_tokens, pad_array))
        assert len(imdb[i]['question_tokens']) == max_len, len(imdb[i]['question_tokens'])


def build_imdb(image_set):
    global max_len

    print('building imdb %s' % image_set)
    load_answer = True
    assert image_set in ['train', 'dev', 'test1']
    with open(examples_file.format(set=image_set)) as f:
        examples = [json.loads(line) for line in f if line]

    vocab_dict = text_processing.VocabDict(question_vocab_file)

    imdb = []

    image_pair_regex = re.compile(r'^[-\w]+(?=-\d)')
    dset_max_len = 0

    for example in tqdm(examples, file=sys.stdout):
        question_id = example['identifier']

        image_id = image_pair_regex.match(question_id).group(0)
        images_dir = os.path.abspath(os.path.join(image_dir % image_set))
        left_image_path = os.path.join(images_dir, f'{image_id}-img0.png')
        right_image_path = os.path.join(images_dir, f'{image_id}-img1.png')
        feature_path = os.path.abspath(os.path.join(feature_dir % image_set, f'{image_id}.npy'))

        missing_images = [not os.path.isfile(img) for img in [left_image_path, right_image_path]]
        if any(missing_images):
            tqdm.write("Image Missing:\t" +
                       ", ".join(["left", "right"][i] for i in range(len(missing_images)) if missing_images[i]))
            continue

        if not os.path.isfile(feature_path):
            tqdm.write("Image Features Missing:\t" + feature_path)
            continue

        question_str = example['sentence']
        # question_tokens = text_processing.tokenize(question_str)
        question_tokens = np.array([vocab_dict.word2idx(w) for w in text_processing.tokenize(question_str)])

        if len(question_tokens) > dset_max_len:
            dset_max_len = len(question_tokens)

        iminfo = dict(image_name=image_id,
                      image_path=left_image_path,
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

    print("[%s] Max seq length: %d" % (image_set, dset_max_len))

    if dset_max_len > max_len:
        max_len = dset_max_len

    return imdb


imdb_train = build_imdb('train')
imdb_dev = build_imdb('dev')
imdb_test = build_imdb('test1')

pad_imdb(imdb_train, max_len, 'train')
pad_imdb(imdb_dev, max_len, 'dev')
pad_imdb(imdb_test, max_len, 'test1')

os.makedirs('./imdb_r152_7x7', exist_ok=True)
np.save('./imdb_r152_7x7/imdb_train.npy', np.array(imdb_train))
np.save('./imdb_r152_7x7/imdb_dev.npy', np.array(imdb_dev))
np.save('./imdb_r152_7x7/imdb_traindev.npy', np.array(imdb_train + imdb_dev))
np.save('./imdb_r152_7x7/imdb_test.npy', np.array(imdb_test))
