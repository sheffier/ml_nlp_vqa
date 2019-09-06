import tensorflow as tf
from tensorflow import convert_to_tensor as to_T

from .config import cfg
from . import controller, nmn, input_unit, output_unit, vis


class Model:
    def __init__(self, input_seq_batch, seq_length_batch, image_feat_batch,
                 num_vocab, num_choices, module_names, dropout_keep_prob,
                 scope='model', reuse=None):
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
            lstm_seq, q_encoding, embed_seq = input_unit.build_input_unit(
                input_seq_batch, seq_length_batch, num_vocab)
            kb_batch_left, kb_batch_right = input_unit.build_kb_batch(image_feat_batch)

            # Controller and NMN
            num_module = len(module_names)
            self.controller = controller.Controller(
                lstm_seq, q_encoding, embed_seq, seq_length_batch, num_module)
            self.c_list = self.controller.c_list
            self.module_logits = self.controller.module_logits
            self.module_probs = self.controller.module_probs
            self.module_prob_list = self.controller.module_prob_list
            self.left_right_probs = self.controller.left_right_probs

            left_module_probs, right_module_probs = (tf.identity(self.module_prob_list) for _ in range(2))
            no_op_index = module_names.index('_NoOp')
            for lri, module_probs in enumerate((left_module_probs, right_module_probs)):
                module_probs *= self.left_right_probs[:, lri]
                module_probs[:, no_op_index] += 1 - tf.reduce_sum(module_probs, axis=1)

            self.left_module_prob_list = left_module_probs
            self.right_module_prob_list = right_module_probs

            self.nmn_left = nmn.NMN(kb_batch_left, self.c_list, module_names, left_module_probs)
            self.nmn_right = nmn.NMN(kb_batch_right, self.c_list, module_names, right_module_probs)

            # Output unit
            if cfg.MODEL.BUILD_VQA:
                self.vqa_scores = output_unit.build_output_unit_vqa(
                    q_encoding, (self.nmn_left.mem_last, self.nmn_right.mem_last), num_choices,
                    dropout_keep_prob=dropout_keep_prob)
            if cfg.MODEL.BUILD_LOC:
                raise NotImplementedError
                loc_scores, bbox_offset, bbox_offset_fcn = \
                    output_unit.build_output_unit_loc(
                        q_encoding, kb_batch, self.nmn.att_last)
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
            if cfg.MODEL.BUILD_VQA:
                self.vis_outputs['vqa_scores'] = self.vqa_scores
            if cfg.MODEL.BUILD_LOC:
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
