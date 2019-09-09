from comet_ml import Experiment
import argparse
import os
import numpy as np
import tensorflow as tf

from models_nlvr.model import TrainingModel
from models_nlvr.config import (
    cfg, merge_cfg_from_file, merge_cfg_from_list, evaluate_final_cfg)
from util.nlvr_train.data_reader import DataReader


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

experiment = Experiment(api_key="wZhhsEAf25MNhISJaDP50GDQg", project_name=cfg.EXP_NAME)
experiment.add_tag("fine-tuning")

hyper_params = {"batch_size": cfg.TRAIN.BATCH_SIZE, "feature_dim": cfg.MODEL.FEAT_DIM}
experiment.log_parameters(hyper_params)

# Start session
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH)))

# Data files
imdb_file = cfg.IMDB_FILE % cfg.TRAIN.SPLIT_VQA
data_reader = DataReader(
    imdb_file, shuffle=True, one_pass=False, batch_size=cfg.TRAIN.BATCH_SIZE,
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
seq_length_batch = tf.placeholder(tf.int32, [None])
image_feat_batch = tf.placeholder(
    tf.float32, [None, cfg.MODEL.H_FEAT, 2 * cfg.MODEL.W_FEAT, cfg.MODEL.FEAT_DIM])
dropout_keep_prob = tf.placeholder(tf.float32, shape=())

model = TrainingModel(input_seq_batch, seq_length_batch, image_feat_batch, num_vocab=num_vocab,
                      num_choices=num_choices, module_names=module_names, dropout_keep_prob=dropout_keep_prob)

# Loss function
if cfg.TRAIN.VQA_USE_SOFT_SCORE:
    soft_score_batch = tf.placeholder(tf.float32, [None, num_choices])
    # Summing, instead of averaging over the choices
    loss_vqa = float(num_choices) * tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=model.out.vqa_scores, labels=soft_score_batch))
else:
    answer_label_batch = tf.placeholder(tf.int32, [None])

    loss_vqa_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model.out.vqa_scores, labels=answer_label_batch)

    loss_vqa_acumm = tf.reduce_sum(loss_vqa_per_sample)
    loss_vqa = tf.reduce_mean(loss_vqa_per_sample)
if cfg.TRAIN.USE_GT_LAYOUT:
    gt_layout_batch = tf.placeholder(tf.int32, [None, None])
    loss_layout = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=model.module_logits, labels=gt_layout_batch))
else:
    loss_layout = tf.convert_to_tensor(0.)
loss_rec = model.out.rec_loss
loss_train = (loss_vqa * cfg.TRAIN.VQA_LOSS_WEIGHT +
              loss_layout * cfg.TRAIN.LAYOUT_LOSS_WEIGHT +
              loss_rec * cfg.TRAIN.REC_LOSS_WEIGHT)
loss_total = loss_train + cfg.TRAIN.WEIGHT_DECAY * model.l2_reg

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
snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots
# if cfg.TRAIN.START_ITER > 0:
if True:
    # init variables
    sess.run(tf.global_variables_initializer())

    variables_to_restore = model.base_model.get_variable_list()
    pretrain_saver = tf.train.Saver(variables_to_restore, max_to_keep=None)  # keep all snapshots

    pretrain_dir = cfg.TRAIN.SNAPSHOT_DIR % 'pretrain_new_model'
    snapshot_file = os.path.join(pretrain_dir, 'best_val')
    print('resume training from %s' % snapshot_file)
    pretrain_saver.restore(sess, snapshot_file)
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
loss_vqa_ph = tf.placeholder(tf.float32, [])
loss_layout_ph = tf.placeholder(tf.float32, [])
loss_rec_ph = tf.placeholder(tf.float32, [])
accuracy_ph = tf.placeholder(tf.float32, [])
val_accuracy_ph = tf.placeholder(tf.float32, [])
val_loss_vqa_ph = tf.placeholder(tf.float32, [])

summary_trn = []
summary_trn.append(tf.summary.scalar("loss/vqa", loss_vqa_ph))
summary_trn.append(tf.summary.scalar("loss/layout", loss_layout_ph))
summary_trn.append(tf.summary.scalar("loss/rec", loss_rec_ph))
summary_trn.append(tf.summary.scalar("eval/vqa/accuracy", accuracy_ph))
log_step_trn = tf.summary.merge(summary_trn)

summary_val = []
summary_val.append(tf.summary.scalar("loss/val_vqa", val_loss_vqa_ph))
summary_val.append(tf.summary.scalar("eval/vqa/val_accuracy", val_accuracy_ph))
log_val = tf.summary.merge(summary_val)

imdb_val_file = cfg.IMDB_FILE % cfg.VAL.SPLIT_VQA
val_data_reader = DataReader(
    imdb_val_file, shuffle=False, one_pass=True, batch_size=cfg.VAL.BATCH_SIZE,
    vocab_question_file=cfg.VOCAB_QUESTION_FILE, T_encoder=cfg.MODEL.T_ENCODER,
    vocab_answer_file=cfg.VOCAB_ANSWER_FILE, load_gt_layout=False,
    vocab_layout_file=cfg.VOCAB_LAYOUT_FILE, T_decoder=cfg.MODEL.T_CTRL)

n_val_samples = len(val_data_reader.imdb)
n_iters_per_epoch = len(data_reader.imdb) // cfg.TRAIN.BATCH_SIZE
val_avg_loss = 0.
best_val_acc = 0.
best_val_iter = 0

# Run training
avg_accuracy, accuracy_decay = 0., 0.99
for n_batch, batch in enumerate(data_reader.batches()):
    n_iter = n_batch + cfg.TRAIN.START_ITER

    if n_iter >= cfg.TRAIN.MAX_ITER:
        break

    if ((n_iter+1) % n_iters_per_epoch == 0 or
            (n_iter+1) == cfg.TRAIN.MAX_ITER):
        n_samples = 0
        answer_correct = 0
        val_avg_loss = 0.

        for n_val_batch, val_batch in enumerate(val_data_reader.batches()):
            val_feed_dict = {input_seq_batch: val_batch['input_seq_batch'],
                             seq_length_batch: val_batch['seq_length_batch'],
                             image_feat_batch: val_batch['image_feat_batch'],
                             dropout_keep_prob: 1.}
            if cfg.TRAIN.VQA_USE_SOFT_SCORE:
                val_feed_dict[soft_score_batch] = val_batch['soft_score_batch']
            else:
                val_feed_dict[answer_label_batch] = val_batch['answer_label_batch']
            if cfg.TRAIN.USE_GT_LAYOUT:
                val_feed_dict[gt_layout_batch] = val_batch['gt_layout_batch']
            vqa_scores_val, val_loss_vqa = sess.run((model.out.vqa_scores, loss_vqa_acumm), val_feed_dict)

            # compute accuracy
            vqa_labels = val_batch['answer_label_batch']
            vqa_predictions = np.argmax(vqa_scores_val, axis=1)

            n_samples += len(vqa_labels)
            answer_correct += np.sum(vqa_predictions == vqa_labels)
            val_avg_loss += val_loss_vqa * len(vqa_predictions)
        else:
            val_accuracy = answer_correct / n_samples
            val_avg_loss = val_avg_loss / n_samples

            summary = sess.run(log_val, {
                val_loss_vqa_ph: val_avg_loss,
                val_accuracy_ph: val_accuracy})
            log_writer.add_summary(summary, n_iter + 1)

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_val_iter = n_iter + 1
                snapshot_file = os.path.join(snapshot_dir, "best_val")
                snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
                print('snapshot saved to ' + snapshot_file)

            print("[VAL] exp: %s, iter = %d\n\t" % (cfg.EXP_NAME, n_iter + 1) +
                  "loss (vqa) = %f\n\t" % val_avg_loss +
                  "accuracy (cur) = %f\n\t" % val_accuracy +
                  "best accuracy = %f at iter %d" % (best_val_acc, best_val_iter))

            experiment.log_metric("[VAL] loss (vqa)", val_avg_loss, step=n_iter)
            experiment.log_metric("[VAL] accuracy (cur)", val_accuracy, step=n_iter)
            experiment.log_metric("[VAL] best accuracy", best_val_acc, step=n_iter)
            experiment.log_metric("[VAL] best val iter", best_val_iter, step=n_iter)

    feed_dict = {input_seq_batch: batch['input_seq_batch'],
                 seq_length_batch: batch['seq_length_batch'],
                 image_feat_batch: batch['image_feat_batch'],
                 dropout_keep_prob: cfg.TRAIN.DROPOUT_KEEP_PROB}
    if cfg.TRAIN.VQA_USE_SOFT_SCORE:
        feed_dict[soft_score_batch] = batch['soft_score_batch']
    else:
        feed_dict[answer_label_batch] = batch['answer_label_batch']
    if cfg.TRAIN.USE_GT_LAYOUT:
        feed_dict[gt_layout_batch] = batch['gt_layout_batch']
    vqa_scores_val, loss_vqa_val, loss_layout_val, loss_rec_val, _ = sess.run(
        (model.out.vqa_scores, loss_vqa, loss_layout, loss_rec, train_op),
        feed_dict)

    # compute accuracy
    vqa_labels = batch['answer_label_batch']
    vqa_predictions = np.argmax(vqa_scores_val, axis=1)
    accuracy = np.mean(vqa_predictions == vqa_labels)
    avg_accuracy += (1-accuracy_decay) * (accuracy-avg_accuracy)

    # Add to TensorBoard summary
    if (n_iter+1) % cfg.TRAIN.LOG_INTERVAL == 0:
        print("[TRAIN] exp: %s, iter = %d\n\t" % (cfg.EXP_NAME, n_iter+1) +
              "loss (vqa) = %f, loss (layout) = %f, loss (rec) = %f\n\t" % (
                loss_vqa_val, loss_layout_val, loss_rec_val) +
              "accuracy (cur) = %f, accuracy (avg) = %f" % (
                accuracy, avg_accuracy))

        experiment.log_metric("[TRAIN] loss (vqa)", loss_vqa_val, step=n_iter)
        experiment.log_metric("[TRAIN] accuracy (cur)", accuracy, step=n_iter)
        experiment.log_metric("[TRAIN] accuracy (avg)", avg_accuracy, step=n_iter)

        summary = sess.run(log_step_trn, {
            loss_vqa_ph: loss_vqa_val,
            loss_layout_ph: loss_layout_val,
            loss_rec_ph: loss_rec_val,
            accuracy_ph: avg_accuracy})
        log_writer.add_summary(summary, n_iter+1)

    # Save snapshot
    if ((n_iter+1) % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
            (n_iter+1) == cfg.TRAIN.MAX_ITER):
        snapshot_file = os.path.join(snapshot_dir, str(n_iter+1))
        snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
        print('snapshot saved to ' + snapshot_file)
