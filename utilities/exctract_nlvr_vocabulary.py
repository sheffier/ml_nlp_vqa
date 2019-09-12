import os
import json
from util.text_processing import tokenize

NLVR2_DATA_DIR = '../../DATASETS/nlvr/nlvr2/data/'
OUTPUT_FILE = f'../snmn/exp_nlvr/data/vocabulary_nlvr.txt'


def get_nlvr2_tokens():
    tokens = set()
    for sentence in get_nlvr2_sentences():
        tokens.update(tokenize(sentence))

    return tokens


def get_nlvr2_sentences():
    for split in ('train', 'dev', 'test1'):
        with open(os.path.join(NLVR2_DATA_DIR, f'{split}.json'), 'r') as f:
            for line in f.readlines():
                if not line:
                    continue
                sample = json.loads(line)
                # TODO? synset = ' '.split(sample['synset'])
                sentence = sample['sentence'].translate(str.maketrans({'\u203a': '', '\u2019': "'", '\u00e9': 'e'}))
                return sentence


if __name__ == '__main__':
    with open(OUTPUT_FILE, 'w') as f:
        f.writelines(f'{line}\n' for line in sorted(get_nlvr2_tokens()))
