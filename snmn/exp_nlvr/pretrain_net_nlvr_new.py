from comet_ml import Experiment
import argparse
import os
import math
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from models_nlvr.config import (
    cfg, merge_cfg_from_file, merge_cfg_from_list, evaluate_final_cfg)
from models_nlvr.model import PreTrainModel
from util import text_processing
from util.nlvr_train.data_pipeline import (prepare_mask_dataset_iterators, RandomMaskedDataset)


def model_metrics(model):
    # Loss function
    masked_lm_loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model.out.masked_lm_scores, labels=model.out.masked_lm_labels)

    masked_lm_loss_acumm = tf.reduce_sum(masked_lm_loss_per_sample)
    loss_masked_lm = tf.reduce_mean(masked_lm_loss_per_sample)

    loss_total = loss_masked_lm + cfg.TRAIN.WEIGHT_DECAY * model.l2_reg

    solver = tf.train.AdamOptimizer(learning_rate=model.lr)
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

    evaluate_final_cfg()

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

    train_filenames_ph = tf.placeholder(tf.string, shape=[None])
    val_filenames_ph = tf.placeholder(tf.string, shape=[None])

    next_batch_op, training_init_op, validation_init_op = prepare_mask_dataset_iterators(train_filenames_ph,
                                                                                         val_filenames_ph,
                                                                                         batch_size=cfg.TRAIN.BATCH_SIZE)

    train_sampled_ds = RandomMaskedDataset(train_file_pattern, 5, 576)
    val_sampled_ds = RandomMaskedDataset(val_file_pattern, 5, 47)

    model = PreTrainModel(next_batch_op, num_vocab, module_names)

    masked_lm_loss_acumm, loss_masked_lm, train_op = model_metrics(model)

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH))) as sess:
        sess.run(tf.global_variables_initializer())

        # Save snapshot
        snapshot_dir = cfg.TRAIN.SNAPSHOT_DIR % cfg.EXP_NAME
        os.makedirs(snapshot_dir, exist_ok=True)
        base_variables_to_save = model.base_model.get_variable_list()
        base_snapshot_saver = tf.train.Saver(base_variables_to_save, max_to_keep=None)  # keep all snapshots

        out_variables_to_save = model.out.get_variable_list()
        out_snapshot_saver = tf.train.Saver(out_variables_to_save, max_to_keep=None)  # keep all snapshots

        val_best_acc = 0.
        val_best_epoch = 0
        train_loss_vqa = 0.
        train_accuracy = 0.
        train_avg_accuracy = 0.
        accuracy_decay = 0.99

        new_val_avg_loss = 100.
        old_val_avg_loss = 100.
        val_accuracy = 0.

        n_iter = 0
        n_epochs = 100
        lr = cfg.TRAIN.SOLVER.LR

        with tqdm(total=math.ceil((train_sampled_ds.n_shards_per_dup * n_epochs * 150) / 128), file=sys.stdout) as pbar:
            pbar.set_description('[%s]' % cfg.EXP_NAME)
            pbar.set_postfix(iter=n_iter, epoch=0,
                             train_loss=train_loss_vqa, train_acc=train_accuracy,
                             train_acc_avg=train_avg_accuracy,
                             val_loss=0., val_acc_last_epoch=val_accuracy, val_acc_best=val_best_acc)
            for epoch in range(n_epochs):
                sess.run(training_init_op, {train_filenames_ph: next(train_sampled_ds)})

                while True:
                    try:
                        n_iter += 1

                        lm_scores_value, train_loss_vqa, masked_lm_labels_value, _ = sess.run(
                            (model.out.masked_lm_scores, loss_masked_lm, model.out.masked_lm_labels,
                             train_op),
                            {model.lr: lr})

                        # compute accuracy
                        lm_predictions = np.argmax(lm_scores_value, axis=1)
                        train_accuracy = np.mean(lm_predictions == masked_lm_labels_value)
                        train_avg_accuracy += (1 - accuracy_decay) * (train_accuracy - train_avg_accuracy)

                        pbar.update(1)

                        # Add to TensorBoard summary
                        if n_iter % cfg.TRAIN.LOG_INTERVAL == 0:
                            pbar.set_postfix(iter=n_iter, epoch=epoch,
                                             train_loss=train_loss_vqa, train_acc=train_accuracy,
                                             train_acc_avg=train_avg_accuracy,
                                             val_loss=0 if epoch == 0 else new_val_avg_loss,
                                             val_acc_last_epoch=val_accuracy, val_acc_best=val_best_acc)

                            experiment.log_metric("pretrain/train/loss_lm", train_loss_vqa, step=n_iter)
                            experiment.log_metric("pretrain/train/acc", train_accuracy, step=n_iter)
                            experiment.log_metric("pretrain/train/acc_avg", train_avg_accuracy, step=n_iter)

                        if (n_iter % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                                n_iter == cfg.TRAIN.MAX_ITER):
                        try:
                            snapshot_file = os.path.join(snapshot_dir, 'base_' + str(n_iter))
                            base_snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
                            out_snapshot_saver.save(sess, os.path.join(snapshot_dir, 'out_' + str(n_iter)),
                                                    write_meta_graph=False)
                        except Exception as e:
                            print(e.message, e.args)
                        except:
                            print("Could not save iteration snapshot")
                    except tf.errors.OutOfRangeError:
                        break

                # run validation
                sess.run(validation_init_op, {val_filenames_ph: next(val_sampled_ds)})
                n_samples = 0
                answer_correct = 0
                val_avg_loss = 0.
                while True:
                    # As long as the iterator is not empty
                    try:
                        lm_scores_value, val_loss_masked_lm, masked_lm_labels_value = sess.run(
                            (model.out.masked_lm_scores,
                             masked_lm_loss_acumm,
                             model.out.masked_lm_labels),
                            {model.lr: lr})

                        # compute accuracy
                        lm_predictions = np.argmax(lm_scores_value, axis=1)

                        n_samples += len(masked_lm_labels_value)
                        answer_correct += np.sum(lm_predictions == masked_lm_labels_value)
                        val_avg_loss += val_loss_masked_lm
                    except tf.errors.OutOfRangeError:
                        # Update the average loss for the epoch
                        val_accuracy = answer_correct / n_samples
                        old_val_avg_loss = new_val_avg_loss
                        new_val_avg_loss = val_avg_loss / n_samples

                        if val_accuracy > val_best_acc:
                            val_best_acc = val_accuracy
                            val_best_epoch = epoch
	                        try:
	                            snapshot_file = os.path.join(snapshot_dir, "best_val_base")
	                            base_snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
	                            out_snapshot_saver.save(sess, os.path.join(snapshot_dir, "best_val_out"),
	                                                    write_meta_graph=False)
	                        except Exception as e:
	                            print(e.message, e.args)
	                        except:
	                            print("Could not save iteration snapshot")

                        pbar.set_postfix(iter=n_iter, epoch=epoch,
                                         train_loss=train_loss_vqa, train_acc=train_accuracy,
                                         train_acc_avg=train_avg_accuracy,
                                         val_loss=new_val_avg_loss, val_acc_last_epoch=val_accuracy,
                                         val_acc_best=val_best_epoch)

                        experiment.log_metric("pretrain/val/loss_lm", new_val_avg_loss, step=n_iter)
                        experiment.log_metric("pretrain/val/acc_last_epoch", val_accuracy, step=n_iter)
                        experiment.log_metric("pretrain/val/best_acc", val_best_acc, step=n_iter)
                        experiment.log_metric("pretrain/val/best_epoch", val_best_epoch, step=n_iter)

                        break
