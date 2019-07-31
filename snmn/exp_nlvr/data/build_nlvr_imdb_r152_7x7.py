import json
import os
import re
import sys

import numpy as np
from tqdm import tqdm

sys.path.append('../../')  # NOQA
from util import text_processing

examples_file = '../nlvr_dataset/{set}.json'

image_dir = '../nlvr_images/images/%s/'
feature_dir = './resnet152_c5_7x7/%s/'


def build_imdb(image_set):
    print('building imdb %s' % image_set)
    load_answer = True
    assert image_set in ['train', 'dev', 'test']
    with open(examples_file.format(set='test1' if image_set == 'test' else image_set)) as f:
        examples = [json.loads(line) for line in f if line]

    imdb = []

    image_pair_regex = re.compile(r'^[-\w]+(?=-\d)')

    for example in tqdm(examples, file=sys.stdout):
        question_id = example['identifier']

        image_id = image_pair_regex.match(question_id).group(0)
        image_path = os.path.abspath(os.path.join(image_dir % image_set, f'{image_id}.png'))
        feature_path = os.path.abspath(os.path.join(feature_dir % image_set, f'{image_id}.npy'))

        if not os.path.isfile(image_path):
            tqdm.write("Image Missing:\t" + image_path)
            continue

        if not os.path.isfile(feature_path):
            tqdm.write("Image Features Missing:\t" + feature_path)
            continue

        question_str = example['sentence']
        question_tokens = text_processing.tokenize(question_str)

        iminfo = dict(image_name=image_id,
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

    return imdb


imdb_train = build_imdb('train')
imdb_dev = build_imdb('dev')
imdb_test = build_imdb('test')

os.makedirs('./imdb_r152_7x7', exist_ok=True)
np.save('./imdb_r152_7x7/imdb_train.npy', np.array(imdb_train))
np.save('./imdb_r152_7x7/imdb_dev.npy', np.array(imdb_dev))
np.save('./imdb_r152_7x7/imdb_traindev.npy', np.array(imdb_train + imdb_dev))
np.save('./imdb_r152_7x7/imdb_test.npy', np.array(imdb_test))
