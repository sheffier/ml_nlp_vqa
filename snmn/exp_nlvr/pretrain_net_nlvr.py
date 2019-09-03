from comet_ml import Experiment
import os
import sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import tensorflow as tf

from models_nlvr.model import PreTrainModel
from models_nlvr.config import (
    cfg, merge_cfg_from_file, merge_cfg_from_list)
from util.nlvr_train.data_reader import DataReader
from util.cnn import fc_layer as fc

experiment = Experiment(api_key="uXl6JxxbmcanY3sv7C9ECrL59", project_name='ml-nlp-vqa')
experiment.add_tag("pre-training")

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
merge_cfg_from_file(args.cfg)
assert cfg.EXP_NAME == os.path.basename(args.cfg).replace('.yaml', '')
if args.opts:
    merge_cfg_from_list(args.opts)

hyper_params = {"batch_size": cfg.TRAIN.BATCH_SIZE, "feature_dim": cfg.MODEL.FEAT_DIM}
experiment.log_parameters(hyper_params)

# Start session
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH)))

# Data files
imdb_file = cfg.IMDB_FILE % cfg.TRAIN.SPLIT_VQA
data_reader = DataReader(
    imdb_file, shuffle=True, one_pass=False, prefetch_num=25, is_mask_tokens=True, batch_size=cfg.TRAIN.BATCH_SIZE,
    vocab_question_file=cfg.VOCAB_QUESTION_FILE, T_encoder=cfg.MODEL.T_ENCODER,
    vocab_answer_file=cfg.VOCAB_ANSWER_FILE,
    load_gt_layout=cfg.TRAIN.USE_GT_LAYOUT,
    vocab_layout_file=cfg.VOCAB_LAYOUT_FILE, T_decoder=cfg.MODEL.T_CTRL,
    load_soft_score=cfg.TRAIN.VQA_USE_SOFT_SCORE)
num_vocab = data_reader.batch_loader.vocab_dict.num_vocab
num_choices = data_reader.batch_loader.answer_dict.num_vocab
module_names = data_reader.batch_loader.layout_dict.word_list

# Inputs and model
input_seq_batch = tf.placeholder(tf.int32, [None, None])
output_seq_batch = tf.placeholder(tf.int32, [None, None])
seq_length_batch = tf.placeholder(tf.int32, [None])
image_feat_batch = tf.placeholder(
    tf.float32, [None, cfg.MODEL.H_FEAT, cfg.MODEL.W_FEAT, cfg.MODEL.FEAT_DIM])
dropout_keep_prob = tf.placeholder(tf.float32, shape=())

model = PreTrainModel(input_seq_batch, output_seq_batch, seq_length_batch, image_feat_batch, num_vocab,
                      module_names)

# Loss function
masked_lm_loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=model.out.masked_lm_scores, labels=model.out.masked_lm_labels)

masked_lm_loss_acumm = tf.reduce_sum(masked_lm_loss_per_sample)
loss_masked_lm = tf.reduce_mean(masked_lm_loss_per_sample)
# loss_masked_lm = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
#     logits=model.masked_lm_scores, labels=model.masked_lm_labels))

loss_total = loss_masked_lm + cfg.TRAIN.WEIGHT_DECAY * model.l2_reg

# Train with Adam
solver = tf.train.AdamOptimizer(learning_rate=cfg.TRAIN.SOLVER.LR)
solver_op = solver.minimize(loss_total)
# Save moving average of parameters
ema = tf.train.ExponentialMovingAverage(decay=cfg.TRAIN.EMV_DECAY)
ema_op = ema.apply(model.params)
with tf.control_dependencies([solver_op]):
    train_op = tf.group(ema_op)

# Save snapshot
snapshot_dir = cfg.TRAIN.SNAPSHOT_DIR % cfg.EXP_NAME
os.makedirs(snapshot_dir, exist_ok=True)
variables_to_save = model.base_model.get_variable_list()
snapshot_saver = tf.train.Saver(variables_to_save, max_to_keep=None)  # keep all snapshots
if cfg.TRAIN.START_ITER > 0:
    snapshot_file = os.path.join(snapshot_dir, str(cfg.TRAIN.START_ITER))
    print('resume training from %s' % snapshot_file)
    snapshot_saver.restore(sess, snapshot_file)
else:
    sess.run(tf.global_variables_initializer())
    if cfg.TRAIN.INIT_FROM_WEIGHTS:
        snapshot_saver.restore(sess, cfg.TRAIN.INIT_WEIGHTS_FILE)
        print('initialized from %s' % cfg.TRAIN.INIT_WEIGHTS_FILE)
# Save config
np.save(os.path.join(snapshot_dir, 'cfg.npy'), np.array(cfg))

# Write summary to TensorBoard
log_dir = cfg.TRAIN.LOG_DIR % cfg.EXP_NAME
os.makedirs(log_dir, exist_ok=True)
log_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
loss_masked_lm_ph = tf.placeholder(tf.float32, [])
accuracy_ph = tf.placeholder(tf.float32, [])
val_accuracy_ph = tf.placeholder(tf.float32, [])
val_loss_masked_lm_ph = tf.placeholder(tf.float32, [])

summary_trn = []
summary_trn.append(tf.summary.scalar("loss/lm_train", loss_masked_lm_ph))
summary_trn.append(tf.summary.scalar("accuracy/train", accuracy_ph))
log_step_trn = tf.summary.merge(summary_trn)

summary_val = []
summary_val.append(tf.summary.scalar("loss/lm_validation", val_loss_masked_lm_ph))
summary_val.append(tf.summary.scalar("accuracy/validation", val_accuracy_ph))
log_val = tf.summary.merge(summary_val)

imdb_val_file = cfg.IMDB_FILE % cfg.VAL.SPLIT_VQA
val_data_reader = DataReader(
    imdb_val_file, shuffle=False, one_pass=True, is_mask_tokens=True, batch_size=cfg.VAL.BATCH_SIZE,
    vocab_question_file=cfg.VOCAB_QUESTION_FILE, T_encoder=cfg.MODEL.T_ENCODER,
    vocab_answer_file=cfg.VOCAB_ANSWER_FILE, load_gt_layout=False,
    vocab_layout_file=cfg.VOCAB_LAYOUT_FILE, T_decoder=cfg.MODEL.T_CTRL)

n_val_samples = len(val_data_reader.imdb)
n_iters_per_epoch = len(data_reader.imdb) // cfg.TRAIN.BATCH_SIZE
best_val_acc = 0.
best_val_iter = 0

# Run training
avg_accuracy, accuracy_decay = 0., 0.99
for n_batch, batch in enumerate(data_reader.batches()):
    n_iter = n_batch + cfg.TRAIN.START_ITER
    if n_iter >= cfg.TRAIN.MAX_ITER:
        break

    if ((n_iter+1) % n_iters_per_epoch == 0 or
            (n_iter+1) == cfg.TRAIN.MAX_ITER) or (n_iter == 0):
        n_samples = 0
        answer_correct = 0
        val_avg_loss = 0.

        for n_val_batch, val_batch in enumerate(val_data_reader.batches()):
            val_feed_dict = {input_seq_batch: val_batch['input_seq_batch'],
                             output_seq_batch: val_batch['output_seq_batch'],
                             seq_length_batch: val_batch['seq_length_batch'],
                             image_feat_batch: val_batch['image_feat_batch'],
                             dropout_keep_prob: 1.}
            # if cfg.TRAIN.VQA_USE_SOFT_SCORE:
            #     val_feed_dict[soft_score_batch] = val_batch['soft_score_batch']
            # else:
            #     val_feed_dict[answer_label_batch] = val_batch['answer_label_batch']
            # if cfg.TRAIN.USE_GT_LAYOUT:
            #     val_feed_dict[gt_layout_batch] = val_batch['gt_layout_batch']
            lm_scores_value, val_loss_masked_lm, masked_lm_labels_value = sess.run((model.out.masked_lm_scores,
                                                                                    masked_lm_loss_acumm,
                                                                                    model.out.masked_lm_labels),
                                                                                   val_feed_dict)

            # compute accuracy
            lm_predictions = np.argmax(lm_scores_value, axis=1)

            n_samples += len(masked_lm_labels_value)
            answer_correct += np.sum(lm_predictions == masked_lm_labels_value)
            val_avg_loss += val_loss_masked_lm
        else:
            val_accuracy = answer_correct / n_samples
            val_avg_loss = val_avg_loss / n_samples

            summary = sess.run(log_val, {
                val_loss_masked_lm_ph: val_avg_loss,
                val_accuracy_ph: val_accuracy})
            log_writer.add_summary(summary, n_iter + 1)

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_val_iter = n_iter + 1
                snapshot_file = os.path.join(snapshot_dir, "best_val")
                snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
                print('snapshot saved to ' + snapshot_file)

            print("[VAL] exp: %s, iter = %d\n\t" % (cfg.EXP_NAME, n_iter + 1) +
                  "loss (lm) = %f\n\t" % val_avg_loss +
                  "accuracy (cur) = %f\n\t" % val_accuracy +
                  "best accuracy = %f at iter %d" % (best_val_acc, best_val_iter))

            experiment.log_metric("[VAL] loss (vqa)", val_avg_loss, step=n_iter)
            experiment.log_metric("[VAL] accuracy (cur)", val_accuracy, step=n_iter)
            experiment.log_metric("[VAL] best accuracy", best_val_acc, step=n_iter)
            experiment.log_metric("[VAL] best val iter", best_val_iter, step=n_iter)

    feed_dict = {input_seq_batch: batch['input_seq_batch'],
                 output_seq_batch: batch['output_seq_batch'],
                 seq_length_batch: batch['seq_length_batch'],
                 image_feat_batch: batch['image_feat_batch'],
                 dropout_keep_prob: cfg.TRAIN.DROPOUT_KEEP_PROB}
    # if cfg.TRAIN.VQA_USE_SOFT_SCORE:
    #     feed_dict[soft_score_batch] = batch['soft_score_batch']
    # else:
    #     feed_dict[answer_label_batch] = batch['answer_label_batch']
    # if cfg.TRAIN.USE_GT_LAYOUT:
    #     feed_dict[gt_layout_batch] = batch['gt_layout_batch']

    # lm_scores_value, loss_masked_lm_value, masked_lm_labels_value, _, kk = sess.run(
    #     (model.masked_lm_scores, loss_masked_lm, model.masked_lm_labels, train_op, model.indices),
    #     feed_dict)
    lm_scores_value, loss_masked_lm_value, masked_lm_labels_value, indices_val, _ = sess.run(
        (model.out.masked_lm_scores, loss_masked_lm, model.out.masked_lm_labels, model.out.indices, train_op),
        feed_dict)

    # compute accuracy
    lm_predictions = np.argmax(lm_scores_value, axis=1)
    accuracy = np.mean(lm_predictions == masked_lm_labels_value)
    avg_accuracy += (1-accuracy_decay) * (accuracy-avg_accuracy)

    # Add to TensorBoard summary
    if (n_iter+1) % cfg.TRAIN.LOG_INTERVAL == 0:
        print("[TRAIN] exp: %s, iter = %d\n\t" % (cfg.EXP_NAME, n_iter+1) +
              "loss (lm) = %f\n\t" % (
                loss_masked_lm_value) +
              "accuracy (cur) = %f, accuracy (avg) = %f" % (
                accuracy, avg_accuracy))

        experiment.log_metric("[TRAIN] loss (vqa)", loss_masked_lm_value, step=n_iter)
        experiment.log_metric("[TRAIN] accuracy (cur)", accuracy, step=n_iter)
        experiment.log_metric("[TRAIN] accuracy (avg)", avg_accuracy, step=n_iter)
        experiment.log_metrics({'n_indices': len(indices_val), 'n_samples': np.sum(batch['seq_length_batch'])},
                               step=n_iter)

        summary = sess.run(log_step_trn, {
            loss_masked_lm_ph: loss_masked_lm_value,
            accuracy_ph: avg_accuracy})
        log_writer.add_summary(summary, n_iter+1)

    # Save snapshot
    if ((n_iter+1) % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
            (n_iter+1) == cfg.TRAIN.MAX_ITER):
        snapshot_file = os.path.join(snapshot_dir, str(n_iter+1))
        snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
        print('snapshot saved to ' + snapshot_file)
