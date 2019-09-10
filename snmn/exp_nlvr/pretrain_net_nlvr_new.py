from comet_ml import Experiment
import argparse
import math
import os

import numpy as np
import tensorflow as tf
from models_nlvr.config import (
    cfg, merge_cfg_from_file, merge_cfg_from_list, evaluate_final_cfg)
from models_nlvr.model import PreTrainModel
from tqdm import tqdm
from util import (text_processing, session)
from util.nlvr_train.data_pipeline import (prepare_mask_dataset_iterators, RandomMaskedDataset)


class Stats:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    merge_cfg_from_file(args.cfg)
    assert os.path.basename(args.cfg).replace('.yaml', '') in cfg.EXP_NAME
    if args.opts:
        merge_cfg_from_list(args.opts)

    evaluate_final_cfg()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)

    experiment = Experiment(api_key="uXl6JxxbmcanY3sv7C9ECrL59", project_name='ml-nlp-vqa')
    experiment.add_tag("pretraining-new-model")

    hyper_params = {"batch_size": cfg.TRAIN.BATCH_SIZE, "feature_dim": cfg.MODEL.FEAT_DIM}
    experiment.log_parameters(hyper_params)

    dataset_dir = './exp_nlvr/data/tfrecords/'
    qst_vocab_file = cfg.VOCAB_QUESTION_FILE
    ans_vocab_file = cfg.VOCAB_ANSWER_FILE
    layout_vocab_file = cfg.VOCAB_LAYOUT_FILE

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

    masked_lm_loss_acumm, loss_masked_lm, train_op = model.get_metrics()

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH))) as sess:
        session.init_session(sess, model)

        # Save snapshot
        snapshot_dir = cfg.TRAIN.SNAPSHOT_DIR % cfg.EXP_NAME
        os.makedirs(snapshot_dir, exist_ok=True)
        base_variables_to_save = model.base_model.get_variable_list()
        base_snapshot_saver = tf.train.Saver(base_variables_to_save, max_to_keep=None)  # keep all snapshots

        out_variables_to_save = model.out.get_variable_list()
        out_snapshot_saver = tf.train.Saver(out_variables_to_save, max_to_keep=None)  # keep all snapshots

        n_iter = 0
        n_epochs = 100
        lr = cfg.TRAIN.SOLVER.LR
        accuracy_decay = 0.99

        train_stats = Stats(loss=0., acc=0., avg_acc=0.)
        val_stats = Stats(loss=0., last_acc=0., best_acc=0., best_epoch=0)

        train_log = tqdm(position=0, bar_format='{desc}')
        val_log = tqdm(position=1, bar_format='{desc}')
        train_pbar = tqdm(total=math.ceil((train_sampled_ds.n_shards_per_dup * n_epochs * 150) / 128),
                          desc=f'[{cfg.EXP_NAME}]', position=2)


        def update_train_log(t_stats):
            train_log.set_description_str(f'[TRAIN_STATS] loss={t_stats.loss:2.4f}'
                                          f' acc={t_stats.acc:2.4f} '
                                          f' avg_acc={t_stats.avg_acc:2.4f}')


        def update_val_log(v_stats):
            val_log.set_description_str(f'[VAL_STATS] loss={v_stats.loss:2.4f}'
                                        f' last_epoch_acc={v_stats.last_acc:2.4f}'
                                        f' best_acc={v_stats.best_acc:2.4f} @epoch {v_stats.best_epoch:d}')


        update_train_log(train_stats)
        update_val_log(val_stats)

        for epoch in range(n_epochs):
            sess.run(training_init_op, {train_filenames_ph: next(train_sampled_ds)})
            train_pbar.set_postfix(epoch=epoch)
            while True:
                try:
                    n_iter += 1

                    lm_scores_value, train_stats.loss, masked_lm_labels_value, _ = sess.run(
                        (model.out.masked_lm_scores, loss_masked_lm, model.out.masked_lm_labels,
                         train_op),
                        {model.lr: lr})

                    # compute accuracy
                    lm_predictions = np.argmax(lm_scores_value, axis=1)
                    train_stats.acc = np.mean(lm_predictions == masked_lm_labels_value)
                    train_stats.avg_acc += (1 - accuracy_decay) * (train_stats.acc - train_stats.avg_acc)

                    update_train_log(train_stats)
                    val_log.refresh()
                    train_pbar.update(1)

                    # Add to TensorBoard summary
                    if n_iter % cfg.TRAIN.LOG_INTERVAL == 0:
                        experiment.log_metric("pretrain/train/loss_lm", train_stats.loss, step=n_iter)
                        experiment.log_metric("pretrain/train/acc", train_stats.acc, step=n_iter)
                        experiment.log_metric("pretrain/train/acc_avg", train_stats.avg_acc, step=n_iter)

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
            val_pbar = tqdm(total=math.ceil((val_sampled_ds.n_shards_per_dup * 150) / 128),
                            desc='[VAL]', position=3)

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

                    val_pbar.update(1)
                    train_log.refresh()
                    val_log.refresh()
                except tf.errors.OutOfRangeError:
                    # Update the average loss for the epoch
                    val_stats.last_acc = answer_correct / n_samples
                    val_stats.loss = val_avg_loss / n_samples

                    if val_stats.last_acc > val_stats.best_acc:
                        val_stats.best_acc = val_stats.last_acc
                        val_stats.best_epoch = epoch

                        try:
                            snapshot_file = os.path.join(snapshot_dir, "best_val_base")
                            base_snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
                            out_snapshot_saver.save(sess, os.path.join(snapshot_dir, "best_val_out"),
                                                    write_meta_graph=False)
                        except Exception as e:
                            print(e.message, e.args)
                        except:
                            print("Could not save iteration snapshot")

                    update_val_log(val_stats)

                    experiment.log_metric("pretrain/val/loss_lm", val_stats.loss, step=n_iter)
                    experiment.log_metric("pretrain/val/acc_last_epoch", val_stats.last_acc, step=n_iter)
                    experiment.log_metric("pretrain/val/best_acc", val_stats.best_acc, step=n_iter)
                    experiment.log_metric("pretrain/val/best_epoch", val_stats.best_epoch, step=n_iter)

                    break

            val_pbar.close()

        train_pbar.close()
        train_log.close()
        val_log.close()
