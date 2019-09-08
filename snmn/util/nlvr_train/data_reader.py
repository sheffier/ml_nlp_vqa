import threading
import queue
import random
import numpy as np

from util import text_processing


class BatchLoaderVqa:
    def __init__(self, imdb, is_mask_tokens, data_params):
        self.imdb = imdb
        self.is_mask_tokens = is_mask_tokens
        self.data_params = data_params

        self.vocab_dict = text_processing.VocabDict(
            data_params['vocab_question_file'])
        self.T_encoder = data_params['T_encoder']

        # peek one example to see whether answer and gt_layout are in the data
        self.load_answer = 'answer' in self.imdb[0]
        self.load_gt_layout = False
        # the answer dict is always loaded, regardless of self.load_answer
        self.answer_dict = text_processing.VocabDict(data_params['vocab_answer_file'])
        if not self.load_answer:
            print('imdb does not contain answers')
        self.T_decoder = data_params['T_decoder']
        self.layout_dict = text_processing.VocabDict(data_params['vocab_layout_file'])
        if self.load_gt_layout:
            # Prune multiple filter modules by default
            self.prune_filter_module = (
                data_params['prune_filter_module']
                if 'prune_filter_module' in data_params else True)
        else:
            print('imdb does not contain ground-truth layout')
        # Whether to load soft scores (targets for sigmoid regression)
        self.load_soft_score = False

        # load one feature map to peek its size
        feats = np.load(self.imdb[0]['feature_path'], allow_pickle=True)
        self.feat_H, self.feat_W, self.feat_D = feats.shape[1:]

    @staticmethod
    def random_word(tokens, out_labels, vocab_dict):
        # input_seq = []
        # output_label = []

        for i, token in enumerate(tokens):
            # token_idx = vocab_dict.word2idx(token)
            prob = random.random()
            # mask token with probability
            ratio = 0.15  # args.word_mask_rate
            if prob < ratio:
                prob /= ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    # input_seq.append(vocab_dict.MASK_idx)
                    tokens[i] = vocab_dict.MASK_idx

                # 10% randomly change token to random token
                elif prob < 0.9:
                    # input_seq.append(random.randrange(vocab_dict.num_vocab + 1))
                    tokens[i] = random.randrange(vocab_dict.num_vocab + 1)
                # else: # -> rest 10% randomly keep current token
                #     input_seq.append(token_idx)

                # out_labels[i] = token_idx
                out_labels[i] = token
                # output_label.append(token_idx)
            # else:
            #     # no masking token (will be ignored by loss function later)
            #     input_seq.append(token_idx)
            #     # output_label.append(-1)

        # return input_seq  # , output_label

    def load_one_batch(self, sample_ids):
        actual_batch_size = len(sample_ids)
        input_seq_batch = np.empty(
            (self.T_encoder, actual_batch_size), np.int32)
        if self.is_mask_tokens:
            output_seq_batch = np.ones(
                (self.T_encoder, actual_batch_size), np.int32) * -1
        seq_length_batch = np.empty(actual_batch_size, np.int32)
        image_feat_batch = np.empty(
            (actual_batch_size, self.feat_H, self.feat_W, self.feat_D),
            np.float32)
        if self.load_answer:
            answer_label_batch = np.empty(actual_batch_size, np.int32)
            # valid_answers_list = [None]*actual_batch_size
            # all_answers_list = [None]*actual_batch_size
            if self.load_soft_score:
                num_choices = len(self.answer_dict.word_list)
                soft_score_batch = np.zeros(
                    (actual_batch_size, num_choices), np.float32)
        if self.load_gt_layout:
            gt_layout_batch = self.layout_dict.word2idx('_NoOp') * np.ones(
                (self.T_decoder, actual_batch_size), np.int32)

        for n in range(len(sample_ids)):
            iminfo = self.imdb[sample_ids[n]]
            seq_length = len(iminfo['question_tokens'])
            seq_length_batch[n] = seq_length
            image_feat_batch[n:n+1] = np.load(iminfo['feature_path'], allow_pickle=True)
            if self.load_answer:
                answer_idx = self.answer_dict.word2idx(iminfo['answer'])
                answer_label_batch[n] = answer_idx
            if self.load_gt_layout:
                gt_layout_tokens = iminfo['gt_layout_tokens']
                if self.prune_filter_module:
                    # remove duplicated consecutive modules
                    # (only keeping one _Filter)
                    for n_t in range(len(gt_layout_tokens)-1, 0, -1):
                        if (gt_layout_tokens[n_t-1] in {'_Filter', '_Find'}
                                and gt_layout_tokens[n_t] == '_Filter'):
                            gt_layout_tokens[n_t] = None
                    gt_layout_tokens = [t for t in gt_layout_tokens if t]
                layout_inds = [
                    self.layout_dict.word2idx(w) for w in gt_layout_tokens]
                gt_layout_batch[:len(layout_inds), n] = layout_inds

            input_seq_batch[:, n] = iminfo['question_tokens']
            if self.is_mask_tokens:
                # question_inds, question_labels = self.random_word(iminfo['question_tokens'], self.vocab_dict)
                # output_seq_batch[:seq_length, n] = question_labels

                self.random_word(input_seq_batch[:seq_length, n],
                                 output_seq_batch[:seq_length, n],
                                 self.vocab_dict)
            # else:
            #     question_inds = [
            #         self.vocab_dict.word2idx(w) for w in iminfo['question_tokens']]

            # input_seq_batch[:seq_length, n] = question_inds

        batch = dict(input_seq_batch=input_seq_batch,
                     seq_length_batch=seq_length_batch,
                     image_feat_batch=image_feat_batch,
                     sample_ids=sample_ids)
        if self.is_mask_tokens:
            batch['output_seq_batch'] = output_seq_batch
        if self.load_answer:
            batch['answer_label_batch'] = answer_label_batch
            # batch['valid_answers_list'] = valid_answers_list
            # batch['all_answers_list'] = all_answers_list
        if self.load_gt_layout:
            batch['gt_layout_batch'] = gt_layout_batch
        return batch


class DataReader:
    def __init__(self, imdb_file, shuffle=True, one_pass=False, prefetch_num=8, is_mask_tokens=False,
                 **kwargs):
        print('Loading imdb from %s' % imdb_file)
        if imdb_file.endswith('.npy'):
            imdb = np.load(imdb_file, allow_pickle=True)
        else:
            raise TypeError('unknown imdb format.')
        print('Done')
        self.imdb = imdb
        self.shuffle = shuffle
        self.one_pass = one_pass
        self.prefetch_num = prefetch_num
        self.is_mask_tokens = is_mask_tokens
        self.data_params = kwargs

        # Vqa data loader
        self.batch_loader = BatchLoaderVqa(self.imdb, self.is_mask_tokens, self.data_params)

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=self.prefetch_num)
        self.prefetch_thread = threading.Thread(
            target=_run_prefetch, args=(
                self.prefetch_queue, self.batch_loader, self.imdb,
                self.shuffle, self.one_pass, self.data_params))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def batches(self):
        while True:
            # Get a batch from the prefetching queue
            if self.prefetch_queue.empty():
                print('data reader: waiting for data loading (IO is slow)...')
            batch = self.prefetch_queue.get(block=True)
            if batch is None:
                assert self.one_pass
                print('data reader: one pass finished')
                # raise StopIteration()
                break
            yield batch


def _run_prefetch(prefetch_queue, batch_loader, imdb, shuffle, one_pass,
                  data_params):
    num_samples = len(imdb)
    batch_size = data_params['batch_size']

    n_sample = 0
    fetch_order = np.arange(num_samples)
    while True:
        # Shuffle the sample order for every epoch
        if n_sample == 0 and shuffle:
            fetch_order = np.random.permutation(num_samples)

        # Load batch from file
        # note that len(sample_ids) <= batch_size, not necessarily equal
        sample_ids = fetch_order[n_sample:n_sample+batch_size]
        batch = batch_loader.load_one_batch(sample_ids)
        prefetch_queue.put(batch, block=True)

        n_sample += len(sample_ids)
        if n_sample >= num_samples:
            # Put in a None batch to indicate a whole pass is over
            if one_pass:
                prefetch_queue.put(None, block=True)
            n_sample = 0
