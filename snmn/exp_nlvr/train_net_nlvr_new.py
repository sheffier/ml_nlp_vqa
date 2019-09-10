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
from models_nlvr.model import TrainingModel
from util import (text_processing, session)
from util.nlvr_train.data_pipeline import prepare_dataset_iterators


def model_metrics(model):
    # Loss function
    loss_vqa_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model.out.vqa_scores, labels=model.answer_batch)

    loss_vqa_acumm = tf.reduce_sum(loss_vqa_per_sample)
    loss_vqa = tf.reduce_mean(loss_vqa_per_sample)

    if cfg.TRAIN.USE_GT_LAYOUT:
        gt_layout_batch = tf.placeholder(tf.int32, [None, None])
        loss_layout = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model.base_model.module_logits, labels=gt_layout_batch))
    else:
        loss_layout = tf.convert_to_tensor(0.)
    loss_rec = model.out.rec_loss
    loss_train = (loss_vqa * cfg.TRAIN.VQA_LOSS_WEIGHT +
                  loss_layout * cfg.TRAIN.LAYOUT_LOSS_WEIGHT +
                  loss_rec * cfg.TRAIN.REC_LOSS_WEIGHT)
    loss_total = loss_train + cfg.TRAIN.WEIGHT_DECAY * model.l2_reg

    solver = tf.train.AdamOptimizer(learning_rate=model.lr)
    solver_op = solver.minimize(loss_total)
    # Save moving average of parameters
    ema = tf.train.ExponentialMovingAverage(decay=cfg.TRAIN.EMV_DECAY)
    ema_op = ema.apply(model.params)
    with tf.control_dependencies([solver_op]):
        train_op = tf.group(ema_op)

    return loss_total, loss_vqa, loss_vqa_acumm, loss_layout, loss_rec, train_op


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
    experiment.add_tag("training-new-model")

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
    num_choices = ans_vocab_dict.num_vocab

    train_file_pattern = os.path.join(dataset_dir, 'train_*.tfrecord')
    val_file_pattern = os.path.join(dataset_dir, 'dev_*.tfrecord')
    next_batch_op, training_init_op, validation_init_op = prepare_dataset_iterators(train_file_pattern,
                                                                                    val_file_pattern,
                                                                                    batch_size=cfg.TRAIN.BATCH_SIZE)

    model = TrainingModel(next_batch_op, num_vocab, module_names, num_choices)

    loss_total, loss_vqa, loss_vqa_acumm, loss_layout, loss_rec, train_op = model_metrics(model)

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH))) as sess:
        session.init_session(sess)

        # Save snapshot
        snapshot_dir = cfg.TRAIN.SNAPSHOT_DIR % cfg.EXP_NAME
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots

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
        num_epochs = 100
        keep_prob = cfg.TRAIN.DROPOUT_KEEP_PROB
        lr = cfg.TRAIN.SOLVER.LR

        with tqdm(total=math.ceil((len(tf.gfile.Glob(train_file_pattern)) * num_epochs * 150) / 128), file=sys.stdout) as pbar:
            pbar.set_description('[%s]' % cfg.EXP_NAME)
            pbar.set_postfix(iter=n_iter, epoch=0,
                             train_loss=train_loss_vqa, train_acc=train_accuracy,
                             train_acc_avg=train_avg_accuracy,
                             val_loss=0., val_acc_last_epoch=val_accuracy, val_acc_best=val_best_acc)
            for epoch in range(num_epochs):
                sess.run(training_init_op)
                while True:
                    try:
                        n_iter += 1

                        vqa_scores_val, answer_labels, train_loss_vqa, loss_layout_val, loss_rec_val, _ = sess.run(
                            (model.out.vqa_scores, model.answer_batch, loss_vqa, loss_layout, loss_rec, train_op),
                            feed_dict={model.lr: lr, model.dropout_keep_prob: keep_prob})

                        # compute accuracy
                        vqa_predictions = np.argmax(vqa_scores_val, axis=1)
                        train_accuracy = np.mean(vqa_predictions == answer_labels)
                        train_avg_accuracy += (1 - accuracy_decay) * (train_accuracy - train_avg_accuracy)

                        pbar.update(1)

                        # Add to TensorBoard summary
                        if n_iter % cfg.TRAIN.LOG_INTERVAL == 0:
                            pbar.set_postfix(iter=n_iter, epoch=epoch,
                                             train_loss=train_loss_vqa, train_acc=train_accuracy,
                                             train_acc_avg=train_avg_accuracy,
                                             val_loss=0 if epoch == 0 else new_val_avg_loss,
                                             val_acc_last_epoch=val_accuracy, val_acc_best=val_best_acc)

                            experiment.log_metric("train/loss_vqa", train_loss_vqa, step=n_iter)
                            experiment.log_metric("train/acc", train_accuracy, step=n_iter)
                            experiment.log_metric("train/avg_acc", train_accuracy, step=n_iter)

                        if (n_iter % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                                n_iter == cfg.TRAIN.MAX_ITER):
                            try:
                                snapshot_file = os.path.join(snapshot_dir, str(n_iter))
                                snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
                            except Exception as e:
                                print(e.message, e.args)
                            except:
                                print("Could not save iteration snapshot")
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
                        vqa_scores_val, answer_labels, val_loss_vqa = \
                            sess.run((model.out.vqa_scores, model.answer_batch, loss_vqa_acumm),
                                     feed_dict={model.lr: lr, model.dropout_keep_prob: 1.})

                        # compute accuracy
                        vqa_predictions = np.argmax(vqa_scores_val, axis=1)

                        n_samples += len(answer_labels)
                        answer_correct += np.sum(vqa_predictions == answer_labels)
                        val_avg_loss += val_loss_vqa
                    except tf.errors.OutOfRangeError:
                        # Update the average loss for the epoch
                        old_val_avg_loss = new_val_avg_loss
                        val_accuracy = answer_correct / n_samples
                        new_val_avg_loss = val_avg_loss / n_samples

                        if val_accuracy > val_best_acc:
                            val_best_acc = val_accuracy
                            val_best_epoch = epoch

                            try:
                                snapshot_file = os.path.join(snapshot_dir, "best_val")
                                snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
                            except Exception as e:
                                print(e.message, e.args)
                            except:
                                print("Could not save best snapshot")

                        pbar.set_postfix(iter=n_iter, epoch=epoch,
                                         train_loss=train_loss_vqa, train_acc=train_accuracy,
                                         train_acc_avg=train_avg_accuracy,
                                         val_loss=new_val_avg_loss, val_acc_last_epoch=val_accuracy,
                                         val_acc_best=val_best_acc)

                        experiment.log_metric("val/loss_vqa", new_val_avg_loss, step=n_iter)
                        experiment.log_metric("val/acc_last_epoch", val_accuracy, step=n_iter)
                        experiment.log_metric("val/best_acc", val_best_acc, step=n_iter)
                        experiment.log_metric("val/best_epoch", val_best_epoch, step=n_iter)

                        break
