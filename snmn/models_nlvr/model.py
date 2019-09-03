import tensorflow as tf
import random
from tensorflow import convert_to_tensor as to_T, newaxis as ax

from .config import cfg
from util.cnn import fc_layer as fc
from . import controller, nmn, input_unit, output_unit, vis


class PreTrainOutputs:
    def __init__(self, model, output_seq_batch, num_vocab, scope='pretrain_out', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            S = tf.shape(output_seq_batch)[0]
            N = tf.shape(output_seq_batch)[1]

            # (N*H*W , C)
            kb_batch_att = tf.reshape(model.kb_batch * model.nmn.att_last,
                                      (N * cfg.MODEL.H_FEAT * cfg.MODEL.W_FEAT, cfg.MODEL.KB_DIM))

            # (N*H*W , C) -> (N*H*W , d)
            Wk = fc("temp_name__", kb_batch_att, output_dim=cfg.MODEL.LSTM_DIM)
            Wk = tf.reshape(Wk, (-1, cfg.MODEL.H_FEAT * cfg.MODEL.W_FEAT, cfg.MODEL.LSTM_DIM))

            #  j  i   l      i  j  k     j   l   k
            # (N, S, H*W) = (S, N, d) * (N, H*W, d)
            scores = tf.einsum('ijk, jlk->jil', model.lstm_seq, Wk)
            scores = tf.nn.softmax(scores, axis=-1)

            #  i  j  l     i  j   k      i  k    l
            # (N, S, d) = (N, S, H*W) * (N, H*W, d)
            kb_lstm = tf.einsum('ijk, ikl->ijl', scores, Wk)
            kb_lstm = kb_lstm + tf.transpose(model.lstm_seq, perm=[1, 0, 2])  # (N, S, d)
            kb_lstm = tf.reshape(kb_lstm, (N * S, cfg.MODEL.LSTM_DIM))

            out_scores = fc("fc_out_scores", kb_lstm, output_dim=(num_vocab + 1))

            lm_scores = tf.reshape(out_scores, (N, S, (num_vocab + 1)))

            mask_idx = tf.constant([-1])
            self.indices = tf.where(tf.not_equal(tf.transpose(output_seq_batch, perm=[1, 0]), mask_idx))
            self.masked_lm_labels = tf.gather_nd(tf.transpose(output_seq_batch, perm=[1, 0]), self.indices)
            self.masked_lm_scores = tf.gather_nd(lm_scores, self.indices)


class TrainingOutputs:
    def __init__(self, model, input_seq_batch, seq_length_batch, num_vocab, num_choices, dropout_keep_prob,
                 scope='train_out', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if cfg.MODEL.BUILD_VQA:
                self.vqa_scores = output_unit.build_output_unit_vqa(
                    model.q_encoding, model.nmn.mem_last, num_choices,
                    dropout_keep_prob=dropout_keep_prob)
            if cfg.MODEL.BUILD_LOC:
                loc_scores, bbox_offset, bbox_offset_fcn = \
                    output_unit.build_output_unit_loc(
                        model.q_encoding, model.kb_batch, model.nmn.att_last)
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
            self.kb_batch = input_unit.build_kb_batch(image_feat_batch)

            # Controller and NMN
            num_module = len(module_names)
            self.controller = controller.Controller(
                self.lstm_seq, self.q_encoding, self.embed_seq, seq_length_batch, num_module)
            self.c_list = self.controller.c_list
            self.module_logits = self.controller.module_logits
            self.module_probs = self.controller.module_probs
            self.module_prob_list = self.controller.module_prob_list
            self.nmn = nmn.NMN(
                self.kb_batch, self.c_list, module_names, self.module_prob_list)

    def get_variable_list(self):
        vars = tf.trainable_variables(scope=self.scope)

        return vars


class PreTrainModel:
    def __init__(self, input_seq_batch, output_seq_batch, seq_length_batch, image_feat_batch, num_vocab,
                 module_names, scope='full_model', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            self.base_model = BaseModel(input_seq_batch, seq_length_batch, image_feat_batch, num_vocab, module_names,
                                        reuse=reuse)

            self.out = PreTrainOutputs(self.base_model, output_seq_batch, num_vocab, reuse=reuse)

            self.params = [
                v for v in tf.trainable_variables() if scope in v.op.name]
            self.l2_reg = tf.add_n(
                [tf.nn.l2_loss(v) for v in self.params
                 if v.op.name.endswith('weights')])


class TrainingModel:
    def __init__(self, input_seq_batch, seq_length_batch, image_feat_batch, num_vocab, num_choices, module_names,
                 dropout_keep_prob, scope='full_model', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            self.base_model = BaseModel(input_seq_batch, seq_length_batch, image_feat_batch, num_vocab, module_names,
                                        reuse=reuse)

            self.out = TrainingOutputs(self.base_model, input_seq_batch, seq_length_batch,
                                       num_vocab, num_choices, dropout_keep_prob)

            self.params = [
                v for v in tf.trainable_variables() if scope in v.op.name]
            self.l2_reg = tf.add_n(
                [tf.nn.l2_loss(v) for v in self.params
                 if v.op.name.endswith('weights')])


class Model:
    def __init__(self, input_seq_batch, seq_length_batch, image_feat_batch,
                 num_vocab, num_choices, module_names, dropout_keep_prob,
                 scope='model', reuse=None, is_pretraining=False):
        """
        Neual Module Networks v4 (the whole model)

        Input:
            input_seq_batch: [S, N], tf.int32
            seq_length_batch: [N], tf.int32
            image_feat_batch: [N, H, W, C], tf.float32
        """

        with tf.variable_scope(scope, reuse=reuse):
            self.T_ctrl = cfg.MODEL.T_CTRL

            # Input unit
            self.lstm_seq, q_encoding, embed_seq = input_unit.build_input_unit(
                input_seq_batch, seq_length_batch, num_vocab)
            self.kb_batch = input_unit.build_kb_batch(image_feat_batch)

            # Controller and NMN
            num_module = len(module_names)
            self.controller = controller.Controller(
                self.lstm_seq, q_encoding, embed_seq, seq_length_batch, num_module)
            self.c_list = self.controller.c_list
            self.module_logits = self.controller.module_logits
            self.module_probs = self.controller.module_probs
            self.module_prob_list = self.controller.module_prob_list
            self.nmn = nmn.NMN(
                self.kb_batch, self.c_list, module_names, self.module_prob_list)

            # Output unit
            if cfg.MODEL.BUILD_VQA:
                self.vqa_scores = output_unit.build_output_unit_vqa(
                    q_encoding, self.nmn.mem_last, num_choices,
                    dropout_keep_prob=dropout_keep_prob)
            if cfg.MODEL.BUILD_LOC:
                loc_scores, bbox_offset, bbox_offset_fcn = \
                    output_unit.build_output_unit_loc(
                        q_encoding, self.kb_batch, self.nmn.att_last)
                self.loc_scores = loc_scores
                self.bbox_offset = bbox_offset
                self.bbox_offset_fcn = bbox_offset_fcn

            # Reconstruction loss
            if cfg.MODEL.REC.USE_REC_LOSS:
                rec_inputs = (self.module_logits if cfg.MODEL.REC.USE_LOGITS
                              else self.module_probs)
                if cfg.MODEL.REC.USE_TXT_ATT:
                    rec_inputs = tf.concat(
                        [rec_inputs, tf.stack(self.c_list)], axis=-1)
                self.rec_loss = output_unit.build_output_unit_rec(
                    rec_inputs, input_seq_batch, embed_seq, seq_length_batch,
                    num_vocab)
            else:
                self.rec_loss = tf.convert_to_tensor(0.)

            self.params = [
                v for v in tf.trainable_variables() if scope in v.op.name]
            self.l2_reg = tf.add_n(
                [tf.nn.l2_loss(v) for v in self.params
                 if v.op.name.endswith('weights')])

            # tensors for visualization
            self.vis_outputs = {
                'txt_att':  # [N, T, S]
                tf.transpose(  # [S, N, T] -> [N, T, S]
                    tf.concat(self.controller.cv_list, axis=2), (1, 2, 0)),
                'att_stack':  # [N, T, H, W, L]
                tf.stack(self.nmn.att_stack_list, axis=1),
                'stack_ptr':  # [N, T, L]
                tf.stack(self.nmn.stack_ptr_list, axis=1),
                'module_prob':  # [N, T, D]
                tf.stack(self.module_prob_list, axis=1)}
            if cfg.MODEL.BUILD_VQA and not is_pretraining:
                self.vis_outputs['vqa_scores'] = self.vqa_scores
            if cfg.MODEL.BUILD_LOC and not is_pretraining:
                self.vis_outputs['loc_scores'] = self.loc_scores
                self.vis_outputs['bbox_offset'] = self.bbox_offset

    def bbox_offset_loss(self, bbox_ind_batch, bbox_offset_batch):
        if cfg.MODEL.BBOX_REG_AS_FCN:
            N = tf.shape(self.bbox_offset_fcn)[0]
            B = tf.shape(self.bbox_offset_fcn)[1]  # B = H*W
            bbox_offset_flat = tf.reshape(self.bbox_offset_fcn, to_T([N*B, 4]))
            slice_inds = tf.range(N) * B + bbox_ind_batch
            bbox_offset_sliced = tf.gather(bbox_offset_flat, slice_inds)
            loss_bbox_offset = tf.reduce_mean(
                tf.squared_difference(bbox_offset_sliced, bbox_offset_batch))
        else:
            loss_bbox_offset = tf.reduce_mean(
                tf.squared_difference(self.bbox_offset, bbox_offset_batch))

        return loss_bbox_offset

    def vis_batch_vqa(self, *args):
        vis.vis_batch_vqa(self, *args)

    def vis_batch_loc(self, *args):
        vis.vis_batch_loc(self, *args)
