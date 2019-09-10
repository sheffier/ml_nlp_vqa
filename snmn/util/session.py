import tensorflow as tf
import sys
sys.path.append('../')  # NOQA

from models_nlvr.config import cfg


def init_session(sess, model):
    sess.run(tf.global_variables_initializer())

    assert isinstance(cfg.TRAIN.INIT_SNAPSHOT_DICT, dict), "Snapshot configuration is not a dictionary"
    if cfg.TRAIN.INIT_SNAPSHOT_DICT:
        snapshot_dict = cfg.TRAIN.INIT_SNAPSHOT_DICT

        if "full" in snapshot_dict:
            assert len(snapshot_dict) == 1

            print('Loading FULL module snapshot from %s' % snapshot_dict["full"])
            full_snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots
            full_snapshot_saver.restore(sess, snapshot_dict["full"])
        else:
            assert len(snapshot_dict) <= 1
            if "base" in snapshot_dict:
                variables_to_restore = model.base_model.get_variable_list()
                base_saver = tf.train.Saver(variables_to_restore, max_to_keep=None)  # keep all snapshots

                print('Loading BASE module snapshot from %s' % snapshot_dict["base"])
                base_saver.restore(sess, snapshot_dict["base"])
            if "out" in snapshot_dict:
                variables_to_restore = model.out.get_variable_list()
                out_saver = tf.train.Saver(variables_to_restore, max_to_keep=None)  # keep all snapshots

                print('Loading OUTPUT module snapshot from %s' % snapshot_dict["out"])
                out_saver.restore(sess, snapshot_dict["base"])
