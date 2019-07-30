import re
import sys
import numpy as np
import json
import os

sys.path.append('../../')  # NOQA
from util import text_processing


examples_file = '../nlvr_dataset/{set}.json'

image_dir = '../nlvr_dataset/images/%s/'
feature_dir = './resnet152_c5_7x7/%s/'



def build_imdb(image_set):
    print('building imdb %s' % image_set)
    load_answer = True
    assert image_set in ['train', 'dev', 'test']
    with open(examples_file.format(set='test1' if image_set == 'test' else image_set)) as f:
        examples = [json.loads(line) for line in f.readlines() if line]

    imdb = [None] * len(examples)

    image_pair_regex = re.compile(r'^[-\w]+(?=-\d)')

    for i, example in enumerate(examples):
        if (i + 1) % 10000 == 0:
            print('processing %d / %d' % (i + 1, len(examples)))

        example_id = example['identifier']

        image_pair_id = image_pair_regex.match(example_id).group(0)
        image_path = os.path.abspath(os.path.join(image_dir % image_set, f'{image_pair_id}.png'))
        feature_path = os.path.abspath(os.path.join(feature_dir % image_set, f'{image_pair_id}.npy'))

        sentence_str = example['sentence']
        sentence_tokens = text_processing.tokenize(sentence_str)

        iminfo = dict(image_path=image_path,
                      image_pair_id=image_pair_id,
                      example_id=example_id,
                      feature_path=feature_path,
                      sentence_str=sentence_str,
                      sentence_tokens=sentence_tokens)

        # load answers
        if load_answer:
            iminfo['answer'] = example['label']  # Assumption: answer is always "True"/"False"

        imdb[i] = iminfo

    return imdb


imdb_train = build_imdb('train')
imdb_dev = build_imdb('dev')
imdb_test = build_imdb('test')

os.makedirs('./imdb_r152_7x7', exist_ok=True)
np.save('./imdb_r152_7x7/imdb_train.npy', np.array(imdb_train))
np.save('./imdb_r152_7x7/imdb_dev.npy', np.array(imdb_dev))
np.save('./imdb_r152_7x7/imdb_test.npy', np.array(imdb_test))
