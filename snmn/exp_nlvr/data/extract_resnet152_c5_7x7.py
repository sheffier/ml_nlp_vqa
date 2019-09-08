import argparse
import os
import sys; sys.path.append('../../')  # NOQA
import re
import subprocess
from glob import glob
import humanfriendly
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf

from util.nets import resnet_v1, channel_mean

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()

gpu_id = args.gpu_id  # set GPU id to use


def choose_gpu(gpu_id):
    NVIDIA_SMI_GPU_RE = re.compile(r"^\|\s*"
                                   r"(?P<id>\d+)\s+TITAN"
                                   r".*\r?\n.*"
                                   r"(?P<mem_usage>\d+\w+)\s*/\s*(?P<total_mem>\d+\w+)"
                                   r"\s*\|\s*"
                                   r"(?P<usage>\d+)%\s+Default"
                                   r"\s*\|\s*$",
                                   re.MULTILINE | re.IGNORECASE)

    if isinstance(gpu_id, (str, bytes)) and gpu_id.lower() == "best":
        nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode('ascii')
        best_gpu_usage = np.inf
        best_available_mem = 0
        best_gpu_id = 0

        def get_size(re_match, group_name):
            return float(humanfriendly.parse_size(re_match.group(group_name)))

        for gpu_match in NVIDIA_SMI_GPU_RE.finditer(nvidia_smi):
            usage = float(gpu_match.group('usage'))
            available_mem = get_size(gpu_match, 'total_mem') - get_size(gpu_match, 'mem_usage')
            if usage < best_gpu_usage or (usage <= best_gpu_usage and available_mem >= best_available_mem):
                best_gpu_id = int(gpu_match.group('id'))
                best_gpu_usage = usage
                best_available_mem = available_mem

        print(f"\x1b[36mGPU CHOSEN AUTOMATICALLY\x1b[0m as {best_gpu_id} (usage "
              f"{best_gpu_usage}%, available memory {humanfriendly.format_size(best_available_mem, binary=True)}).")
        gpu_id = best_gpu_id

    return gpu_id


os.environ['CUDA_VISIBLE_DEVICES'] = str(choose_gpu(gpu_id))

resnet152_model = '../tfmodel/resnet/resnet_v1_152.tfmodel'
image_basedir = '../nlvr_images/images/'
save_basedir = './resnet152_c5_7x7/'
H = 448
W = 448

image_batch = tf.placeholder(tf.float32, [1, H, W, 3])
resnet152_c5 = resnet_v1.resnet_v1_152_c5(image_batch, is_training=False)
resnet152_c5_7x7 = tf.nn.avg_pool(
    resnet152_c5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
saver = tf.train.Saver()
sess = tf.Session(
    config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
saver.restore(sess, resnet152_model)


def extract_image_resnet152_c5_7x7(impath):
    im = skimage.io.imread(impath)
    if im.ndim == 2:  # Gray 2 RGB
        im = np.tile(im[..., np.newaxis], (1, 1, 3))
    im = im[..., :3]
    assert im.dtype == np.uint8
    im = skimage.transform.resize(im, [H, W], preserve_range=True)
    im_val = (im[np.newaxis, ...] - channel_mean)
    resnet152_c5_7x7_val = resnet152_c5_7x7.eval({image_batch: im_val}, sess)
    return resnet152_c5_7x7_val


def extract_dataset_resnet152_c5_7x7(image_dir, save_dir, ext_filter='*-img0.png'):
    image_list = glob(image_dir + '/' + ext_filter)
    os.makedirs(save_dir, exist_ok=True)

    for n_im, impath0 in enumerate(image_list):
        if (n_im+1) % 100 == 0:
            print('processing %d / %d' % (n_im+1, len(image_list)))
        image_name = os.path.basename(impath0).split('.')[0].replace('-img0', '')
        save_path = os.path.join(save_dir, image_name + '.npy')
        if not os.path.exists(save_path):
            resnet152_c5_val0 = extract_image_resnet152_c5_7x7(impath0)
            resnet152_c5_val1 = extract_image_resnet152_c5_7x7(impath0.replace('-img0', '-img1'))
            resnet152_c5_val = np.concatenate([resnet152_c5_val0, resnet152_c5_val1], axis=2)
            np.save(save_path, resnet152_c5_val)


for image_set in ['train', 'dev', 'test1']:
    print('Extracting image set ' + image_set)
    extract_dataset_resnet152_c5_7x7(
        os.path.join(image_basedir, image_set),
        os.path.join(save_basedir, image_set))
    print('Done.')
