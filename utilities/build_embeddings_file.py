from util.text_processing import load_str_list
import tqdm
import numpy as np

VOCAB_FILE = f'../snmn/exp_nlvr/data/vocabulary_nlvr.txt'
EMBEDDINGS_LOOKUP_FILE = f'../../DATASETS/glove/glove.42B.300d.txt'
EMBEDDINGS_FILE = f'../snmn/exp_nlvr/data/vocabulary_nlvr_glove.npy'


def get_words_to_glove_embeddings():
    words_to_glove_embeddings = {}
    with open(EMBEDDINGS_LOOKUP_FILE, 'r') as f:
        for line in tqdm.tqdm(f.readlines()):
            if not line:
                continue
            word, *params = line.split(' ')
            assert len(params) == 300, f'len(params) == {len(params)}'
            words_to_glove_embeddings[word] = np.array(params, dtype=np.float32)

    return words_to_glove_embeddings


if __name__ == '__main__':
    vocab = load_str_list(VOCAB_FILE)
    words_to_glove_embeddings = get_words_to_glove_embeddings()
    vocab = [w if w in words_to_glove_embeddings else '<unk>' for w in vocab]
    embedding_matrix = [
        words_to_glove_embeddings[w]
        for w in vocab
    ]

    np.save(EMBEDDINGS_FILE, np.array(embedding_matrix), allow_pickle=True)
