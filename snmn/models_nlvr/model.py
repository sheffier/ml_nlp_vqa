import tensorflow as tf
from tensorflow import convert_to_tensor as to_T, newaxis as ax
from util.cnn import fc_layer as fc

from . import controller, nmn, input_unit, output_unit, vis
from .config import cfg


def get_shape_list(tensor, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """

    if name is None:
        # TODO(erez) it's currently unused
        name = tensor.name

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = get_shape_list(sequence_tensor)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


class PreTrainOutputs:
    def __init__(self, model, batch_size, max_seq_len, max_pred,
                 masked_lm_ids_batch, masked_lm_positions_batch, masked_lm_weights_batch,
                 num_vocab, scope='pretrain_out', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            self.scope = tf.get_default_graph().get_name_scope()
            S = max_seq_len
            N = batch_size

            # (N*H*W , C)
            left_kb_batch_att = model.kb_batch_left * model.nmn_left.att_last
            right_kb_batch_att = model.kb_batch_right * model.nmn_right.att_last

            kb_batch_att = tf.concat((left_kb_batch_att, right_kb_batch_att), axis=2)
            kb_batch_att = tf.reshape(kb_batch_att, (N * cfg.MODEL.H_FEAT * cfg.MODEL.W_FEAT * 2, cfg.MODEL.KB_DIM))

            # (N*H*W*2 , C) -> (N*H*W*2 , d)
            Wk = fc("temp_name__", kb_batch_att, output_dim=cfg.MODEL.LSTM_DIM)
            Wk = tf.reshape(Wk, (-1, cfg.MODEL.H_FEAT * cfg.MODEL.W_FEAT * 2, cfg.MODEL.LSTM_DIM))

            #  j  i   l      i  j  k     j   l   k
            # (N, S, H*W*2) = (S, N, d) * (N, H*W*2, d)
            scores = tf.einsum('ijk, jlk->jil', model.lstm_seq, Wk)
            scores = tf.nn.softmax(scores, axis=-1)

            #  i  j  l     i  j   k      i  k    l
            # (N, S, d) = (N, S, H*W*2) * (N, H*W*2, d)
            kb_lstm = tf.einsum('ijk, ikl->ijl', scores, Wk)
            kb_lstm = kb_lstm + tf.transpose(model.lstm_seq, perm=[1, 0, 2])  # (N, S, d)
            # kb_lstm = tf.reshape(kb_lstm, (N * S, cfg.MODEL.LSTM_DIM))

            kb_lstm = gather_indexes(kb_lstm, masked_lm_positions_batch)

            out_scores = fc("fc_out_scores", kb_lstm, output_dim=num_vocab)

            mask_idx = tf.constant([1.0])
            t_mask = tf.where(tf.equal(masked_lm_weights_batch, mask_idx))
            self.masked_lm_labels = tf.gather_nd(masked_lm_ids_batch, t_mask)
            self.masked_lm_scores = tf.gather_nd(tf.reshape(out_scores, [N, max_pred, -1]), t_mask)

    def get_variable_list(self):
        vars = tf.trainable_variables(scope=self.scope)

        return vars


class TrainingOutputs:
    def __init__(self, model, input_seq_batch, seq_length_batch, num_vocab, num_choices, dropout_keep_prob,
                 scope='train_out', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if cfg.MODEL.BUILD_VQA:
                self.vqa_scores = output_unit.build_output_unit_vqa(
                    model.q_encoding, model.nmns.mem_last, num_choices,
                    dropout_keep_prob=dropout_keep_prob)
            if cfg.MODEL.BUILD_LOC:
                loc_scores, bbox_offset, bbox_offset_fcn = \
                    output_unit.build_output_unit_loc(
                        model.q_encoding, model.kb_batch, model.nmns.att_last)
                self.loc_scores = loc_scores
                self.bbox_offset = bbox_offset
                self.bbox_offset_fcn = bbox_offset_fcn

            # Reconstruction loss
            if cfg.MODEL.REC.USE_REC_LOSS:
                rec_inputs = (model.module_logits if cfg.MODEL.REC.USE_LOGITS
                              else model.module_probs)
                if cfg.MODEL.REC.USE_TXT_ATT:
                    rec_inputs = tf.concat(
                        [rec_inputs, tf.stack(model.c_list)], axis=-1)
                self.rec_loss = output_unit.build_output_unit_rec(
                    rec_inputs, input_seq_batch, model.embed_seq, seq_length_batch,
                    num_vocab)
            else:
                self.rec_loss = tf.convert_to_tensor(0.)


class BaseModel:
    def __init__(self, input_seq_batch, seq_length_batch, image_feat_batch, num_vocab, module_names,
                 scope='base_model', reuse=None):
        """
        Neual Module Networks v4 (the whole model)

        Input:
            input_seq_batch: [S, N], tf.int32
            seq_length_batch: [N], tf.int32
            image_feat_batch: [N, H, W, C], tf.float32
        """

        with tf.variable_scope(scope, reuse=reuse):
            self.scope = tf.get_default_graph().get_name_scope()
            self.T_ctrl = cfg.MODEL.T_CTRL

            # Input unit
            self.lstm_seq, self.q_encoding, self.embed_seq = input_unit.build_input_unit(
                input_seq_batch, seq_length_batch, num_vocab)
            self.kb_batch_left, self.kb_batch_right = input_unit.build_kb_batch(image_feat_batch)

            # Controller and NMN
            num_module = len(module_names)
            self.controller = controller.Controller(
                self.lstm_seq, self.q_encoding, self.embed_seq, seq_length_batch, num_module)
            self.c_list = self.controller.c_list
            self.module_logits = self.controller.module_logits
            self.module_probs = self.controller.module_probs
            self.module_prob_list = self.controller.module_prob_list
            self.left_right_probs = self.controller.left_right_probs

            left_module_probs, right_module_probs = (tf.identity(self.module_prob_list) for _ in range(2))
            no_op_index = module_names.index('_NoOp')
            for lri, module_probs in enumerate((left_module_probs, right_module_probs)):
                module_probs *= self.left_right_probs[:, :, lri, ax]
                module_probs += (1 - tf.reduce_sum(module_probs, axis=-1, keep_dims=True)
                                 ) * tf.one_hot([[no_op_index]], depth=module_probs.get_shape()[-1])

            self.left_module_prob_list = left_module_probs
            self.right_module_prob_list = right_module_probs

            self.nmn_left = nmn.NMN(self.kb_batch_left, self.c_list, module_names, left_module_probs)
            self.nmn_right = nmn.NMN(self.kb_batch_right, self.c_list, module_names, right_module_probs, reuse=True)
            self.nmns = Aggregate((self.nmn_left, self.nmn_right))

    def get_variable_list(self):
        vars = tf.trainable_variables(scope=self.scope)

        return vars


class PreTrainModel:
    def __init__(self, inputs, num_vocab,
                 module_names, scope='full_model', reuse=None):
        self.lr = tf.placeholder(tf.float32, shape=())
        self.answer_batch = inputs["answer"]
        input_seq_batch = inputs["input_ids"]
        seq_length_batch = inputs["seq_length"]
        masked_lm_positions_batch = inputs["masked_lm_positions"]
        masked_lm_ids_batch = inputs["masked_lm_ids"]
        masked_lm_weights_batch = inputs["masked_lm_weights"]
        image_feat_batch = inputs["img_features"]

        input_seq_batch = tf.transpose(input_seq_batch, perm=[1, 0])

        max_seq_len = tf.shape(input_seq_batch)[0]
        batch_size = tf.shape(input_seq_batch)[1]
        max_pred = tf.shape(masked_lm_weights_batch)[1]

        with tf.variable_scope(scope, reuse=reuse):
            self.base_model = BaseModel(input_seq_batch, seq_length_batch, image_feat_batch, num_vocab, module_names,
                                        reuse=reuse)

            self.out = PreTrainOutputs(self.base_model, batch_size, max_seq_len, max_pred,
                                       masked_lm_ids_batch, masked_lm_positions_batch, masked_lm_weights_batch,
                                       num_vocab, reuse=reuse)

            self.params = [
                v for v in tf.trainable_variables() if scope in v.op.name]
            self.l2_reg = tf.add_n(
                [tf.nn.l2_loss(v) for v in self.params
                 if v.op.name.endswith('weights')])

            self.lengths = seq_length_batch

    def get_metrics(self):
        # Loss function
        masked_lm_loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.out.masked_lm_scores, labels=self.out.masked_lm_labels)

        masked_lm_loss_acumm = tf.reduce_sum(masked_lm_loss_per_sample)
        loss_masked_lm = tf.reduce_mean(masked_lm_loss_per_sample)

        loss_total = loss_masked_lm + cfg.TRAIN.WEIGHT_DECAY * self.l2_reg

        solver = tf.train.AdamOptimizer(learning_rate=self.lr)
        solver_op = solver.minimize(loss_total)
        # Save moving average of parameters
        ema = tf.train.ExponentialMovingAverage(decay=cfg.TRAIN.EMV_DECAY)
        ema_op = ema.apply(self.params)
        with tf.control_dependencies([solver_op]):
            train_op = tf.group(ema_op)

        return masked_lm_loss_acumm, loss_masked_lm, train_op


class TrainingModel:
    def __init__(self, inputs, num_vocab, module_names, num_choices,
                 scope='full_model', reuse=None):
        self.lr = tf.placeholder(tf.float32, shape=())
        self.dropout_keep_prob = tf.placeholder(tf.float32, shape=())

        self.answer_batch = inputs["answer"]
        input_seq_batch = inputs["input_ids"]
        seq_length_batch = inputs["seq_length"]
        image_feat_batch = inputs["img_features"]

        input_seq_batch = tf.transpose(input_seq_batch, perm=[1, 0])

        with tf.variable_scope(scope, reuse=reuse):
            self.base_model = BaseModel(input_seq_batch, seq_length_batch, image_feat_batch, num_vocab, module_names,
                                        reuse=reuse)

            self.out = TrainingOutputs(self.base_model, input_seq_batch, seq_length_batch,
                                       num_vocab, num_choices, self.dropout_keep_prob)

            self.params = [
                v for v in tf.trainable_variables() if scope in v.op.name]
            self.l2_reg = tf.add_n(
                [tf.nn.l2_loss(v) for v in self.params
                 if v.op.name.endswith('weights')])

            # tensors for visualization
            self.vis_outputs = {
                'txt_att':  # [N, T, S]
                tf.transpose(  # [S, N, T] -> [N, T, S]
                    tf.concat(self.base_model.controller.cv_list, axis=2), (1, 2, 0)),
                'att_stack':  # [N, T, H, 2W, L]
                tf.concat(tuple(tf.stack(l, axis=1) for l in self.base_model.nmns.att_stack_list), axis=3),
                'stack_ptr':  # [N, T, L, 2]
                tf.stack(tuple(tf.stack(l, axis=1) for l in self.base_model.nmns.stack_ptr_list), axis=-1),
                'module_prob':  # [N, T, D]
                tf.stack(self.base_model.module_prob_list, axis=1)}
            if cfg.MODEL.BUILD_VQA:
                self.vis_outputs['vqa_scores'] = self.out.vqa_scores
            if cfg.MODEL.BUILD_LOC:
                self.vis_outputs['loc_scores'] = self.out.loc_scores
                self.vis_outputs['bbox_offset'] = self.out.bbox_offset

    def get_metrics(self):
        # Loss function
        loss_vqa_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.out.vqa_scores, labels=self.answer_batch)

        loss_vqa_acumm = tf.reduce_sum(loss_vqa_per_sample)
        loss_vqa = tf.reduce_mean(loss_vqa_per_sample)

        if cfg.TRAIN.USE_GT_LAYOUT:
            gt_layout_batch = tf.placeholder(tf.int32, [None, None])
            loss_layout = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.base_model.module_logits, labels=gt_layout_batch))
        else:
            loss_layout = tf.convert_to_tensor(0.)
        loss_rec = self.out.rec_loss
        loss_train = (loss_vqa * cfg.TRAIN.VQA_LOSS_WEIGHT +
                      loss_layout * cfg.TRAIN.LAYOUT_LOSS_WEIGHT +
                      loss_rec * cfg.TRAIN.REC_LOSS_WEIGHT)
        loss_total = loss_train + cfg.TRAIN.WEIGHT_DECAY * self.l2_reg

        solver = tf.train.AdamOptimizer(learning_rate=self.lr)
        solver_op = solver.minimize(loss_total)
        # Save moving average of parameters
        ema = tf.train.ExponentialMovingAverage(decay=cfg.TRAIN.EMV_DECAY)
        ema_op = ema.apply(self.params)
        with tf.control_dependencies([solver_op]):
            train_op = tf.group(ema_op)

        return loss_total, loss_vqa, loss_vqa_acumm, loss_layout, loss_rec, train_op

    def bbox_offset_loss(self, bbox_ind_batch, bbox_offset_batch):
        if cfg.MODEL.BBOX_REG_AS_FCN:
            N = tf.shape(self.out.bbox_offset_fcn)[0]
            B = tf.shape(self.out.bbox_offset_fcn)[1]  # B = H*W
            bbox_offset_flat = tf.reshape(self.out.bbox_offset_fcn, to_T([N*B, 4]))
            slice_inds = tf.range(N) * B + bbox_ind_batch
            bbox_offset_sliced = tf.gather(bbox_offset_flat, slice_inds)
            loss_bbox_offset = tf.reduce_mean(
                tf.squared_difference(bbox_offset_sliced, bbox_offset_batch))
        else:
            loss_bbox_offset = tf.reduce_mean(
                tf.squared_difference(self.out.bbox_offset, bbox_offset_batch))

        return loss_bbox_offset

    def vis_batch_vqa(self, *args):
        vis.vis_batch_vqa(self, *args)

    def vis_batch_loc(self, *args):
        vis.vis_batch_loc(self, *args)


class Aggregate(tuple):
    def __getattr__(self, item):
        return Aggregate((getattr(self[0], item), getattr(self[1], item)))
