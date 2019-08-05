import os
import sys
import multiprocessing

from tqdm import tqdm
from PIL import Image, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

desired_single_h = 530
desired_single_w = 416


def worker(imageName):
    path1 = os.path.join(inputDir, imageName)
    path2 = path1.replace("img0", "img1")

    try:
        im1 = Image.open(path1)
    except:
        print("ERROR: can not read image\t", path1)
        return

    try:
        im2 = Image.open(path2)
    except:
        print("ERROR: can not read image\t", path2)
        return

    im1 = ResizeSingle(im1, desired_single_h, desired_single_w)
    im2 = ResizeSingle(im2, desired_single_h, desired_single_w)
    im_concat = Image.new('RGB', (2 * desired_single_w, desired_single_h))
    im_concat.paste(im1, (0, 0))
    im_concat.paste(im2, (desired_single_w, 0))
    outPath = os.path.join(outputDir, imageName.replace("-img0", ""))
    # print("Writing\t", imageName.replace("-img0", ""))
    im_concat.save(outPath)


def ResizeSingle(im, h, w):
    old_size = im.size
    ratio = max(float(old_size[0]) / w, float(old_size[1]) / h)
    new_size_single = tuple([int(x / ratio) for x in old_size])  # new_size_single is either (530,x) or (x,416)
    im = im.resize(new_size_single, Image.ANTIALIAS)
    delta_w = w - new_size_single[0]
    delta_h = h - new_size_single[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padding = (left, top, right, bottom)
    im = ImageOps.expand(im, padding)

    return im


if len(sys.argv) != 3:
    print("Should get as args: <input_dir> <output_dir> ")
    exit(1)

inputDir = sys.argv[1]
outputDir = sys.argv[2]

pool = multiprocessing.Pool()
file_list = []

for imageName in os.listdir(inputDir):
    if imageName[-5] != '0':
        continue

    file_list.append(imageName)


with tqdm(total=len(file_list), file=sys.stdout) as t:
    for _ in pool.imap_unordered(worker, file_list):
        t.update(1)
