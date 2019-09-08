from comet_ml import Experiment
import argparse
import os
import math
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from models_nlvr.config import (
    cfg, merge_cfg_from_file, merge_cfg_from_list)
from models_nlvr.model import PreTrainModel
from util import text_processing
from util.nlvr_train.data_pipeline import prepare_dataset_iterators


def model_metrics(model):
    # Loss function
    masked_lm_loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model.out.masked_lm_scores, labels=model.out.masked_lm_labels)

    masked_lm_loss_acumm = tf.reduce_sum(masked_lm_loss_per_sample)
    loss_masked_lm = tf.reduce_mean(masked_lm_loss_per_sample)

    loss_total = loss_masked_lm + cfg.TRAIN.WEIGHT_DECAY * model.l2_reg

    solver = tf.train.AdamOptimizer(learning_rate=cfg.TRAIN.SOLVER.LR)
    solver_op = solver.minimize(loss_total)
    # Save moving average of parameters
    ema = tf.train.ExponentialMovingAverage(decay=cfg.TRAIN.EMV_DECAY)
    ema_op = ema.apply(model.params)
    with tf.control_dependencies([solver_op]):
        train_op = tf.group(ema_op)

    return masked_lm_loss_acumm, loss_masked_lm, train_op


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    merge_cfg_from_file(args.cfg)
    assert cfg.EXP_NAME == os.path.basename(args.cfg).replace('.yaml', '')
    if args.opts:
        merge_cfg_from_list(args.opts)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)

    experiment = Experiment(api_key="uXl6JxxbmcanY3sv7C9ECrL59", project_name='ml-nlp-vqa')
    experiment.add_tag("pretraining-new-model")

    hyper_params = {"batch_size": cfg.TRAIN.BATCH_SIZE, "feature_dim": cfg.MODEL.FEAT_DIM}
    experiment.log_parameters(hyper_params)

    dataset_dir = './exp_nlvr/data/tfrecords/'
    qst_vocab_file = './exp_nlvr/data/vocabulary_nlvr.txt'
    ans_vocab_file = './exp_nlvr/data/answers_nlvr.txt'
    layout_vocab_file = './exp_nlvr/data/vocabulary_layout.txt'

    qst_vocab_dict = text_processing.VocabDict(qst_vocab_file)
    ans_vocab_dict = text_processing.VocabDict(ans_vocab_file)
    layout_vocab_dict = text_processing.VocabDict(layout_vocab_file)

    num_vocab = qst_vocab_dict.num_vocab
    module_names = layout_vocab_dict.word_list

    train_file_pattern = os.path.join(dataset_dir, 'masked_train_*.tfrecord')
    val_file_pattern = os.path.join(dataset_dir, 'masked_dev_*.tfrecord')
    next_batch_op, training_init_op, validation_init_op = prepare_dataset_iterators(train_file_pattern, val_file_pattern)

    model = PreTrainModel(next_batch_op, num_vocab, module_names)

    masked_lm_loss_acumm, loss_masked_lm, train_op = model_metrics(model)

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH))) as sess:
        sess.run(tf.global_variables_initializer())

        # Save snapshot
        snapshot_dir = cfg.TRAIN.SNAPSHOT_DIR % cfg.EXP_NAME
        os.makedirs(snapshot_dir, exist_ok=True)
        variables_to_save = model.base_model.get_variable_list()
        snapshot_saver = tf.train.Saver(variables_to_save, max_to_keep=None)  # keep all snapshots
        epochs = 100
        files = tf.gfile.Glob(train_file_pattern)[0:1]

        avg_accuracy, accuracy_decay = 0., 0.99
        n_iter = 0

        with tqdm(total=math.ceil((len(files) * epochs * 150) / 128), file=sys.stdout) as pbar:
            pbar.set_description('[%s]' % cfg.EXP_NAME)
            for epoch in range(epochs):
                sess.run(training_init_op)

                while True:
                    try:
                        n_iter += 1

                        lm_scores_value, loss_masked_lm_value, masked_lm_labels_value, _ = sess.run(
                            (model.out.masked_lm_scores, loss_masked_lm, model.out.masked_lm_labels,
                             train_op))

                        # compute accuracy
                        lm_predictions = np.argmax(lm_scores_value, axis=1)
                        accuracy = np.mean(lm_predictions == masked_lm_labels_value)
                        avg_accuracy += (1 - accuracy_decay) * (accuracy - avg_accuracy)

                        pbar.update(1)

                        # Add to TensorBoard summary
                        if n_iter % cfg.TRAIN.LOG_INTERVAL == 0:
                            pbar.set_postfix(iter=n_iter, loss=loss_masked_lm_value,
                                             acc=accuracy, avg_acc=avg_accuracy)

                            experiment.log_metric("[PRE-TRAIN] loss (vqa)", loss_masked_lm_value, step=n_iter)
                            experiment.log_metric("[PRE-TRAIN] accuracy (cur)", accuracy, step=n_iter)
                            experiment.log_metric("[PRE-TRAIN] accuracy (avg)", avg_accuracy, step=n_iter)

                        if (n_iter % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                                n_iter == cfg.TRAIN.MAX_ITER):
                            snapshot_file = os.path.join(snapshot_dir, str(n_iter))
                            snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
                            # pbar.set_description('snapshot saved to ' + snapshot_file)

                    except tf.errors.OutOfRangeError:
                        break

                # run validation
                sess.run(validation_init_op)
                n_samples = 0
                answer_correct = 0
                val_avg_loss = 0.
                while True:
                    # As long as the iterator is not empty
                    try:
                        lm_scores_value, val_loss_masked_lm, masked_lm_labels_value = sess.run(
                            (model.out.masked_lm_scores,
                             masked_lm_loss_acumm,
                             model.out.masked_lm_labels))

                        # compute accuracy
                        lm_predictions = np.argmax(lm_scores_value, axis=1)

                        n_samples += len(masked_lm_labels_value)
                        answer_correct += np.sum(lm_predictions == masked_lm_labels_value)
                        val_avg_loss += val_loss_masked_lm
                    except tf.errors.OutOfRangeError:
                        # Update the average loss for the epoch
                        val_accuracy = answer_correct / n_samples
                        val_avg_loss = val_avg_loss / n_samples

                        if val_accuracy > best_val_acc:
                            best_val_acc = val_accuracy
                            best_val_epoch = epoch
                            snapshot_file = os.path.join(snapshot_dir, "best_val")
                            snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)

                        # pbar.set_description('[%s | VAL]' % cfg.EXP_NAME)
                        # pbar.set_postfix(epoch=epoch, loss=val_avg_loss,
                        #                  avg_acc=avg_accuracy, best_acc=best_val_acc)

                        pbar.write("[VAL] epoch = %d, loss %f, acc %f (best_acc %f @epoch %d)" %
                                   (epoch, val_avg_loss, val_accuracy, best_val_acc, best_val_epoch))

                        experiment.log_metric("[VAL] loss (vqa)", val_avg_loss, step=n_iter)
                        experiment.log_metric("[VAL] accuracy (cur)", val_accuracy, step=n_iter)
                        experiment.log_metric("[VAL] best accuracy", best_val_acc, step=n_iter)
                        experiment.log_metric("[VAL] best val epoch", best_val_epoch, step=n_iter)

                        break
