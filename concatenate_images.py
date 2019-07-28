import cv2
import os
import sys


desired_single_h = 530
desired_single_w = 416
desired_concat_size = 448

def ResizeSingle(im, h, w):
    old_size1 = im.shape[:2]
    ratio = max(float(old_size1[0]) / h, float(old_size1[1]) / w)
    new_size_single = tuple([int(x / ratio) for x in old_size1])  # new_size_single is either (530,x) or (x,416)
    im = cv2.resize(im, (new_size_single[1], new_size_single[0]))
    delta_w = w - new_size_single[1]
    delta_h = h - new_size_single[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                            value=[0, 0, 0])
    return im

if len(sys.argv)!=3:
    print("Should get as args: <input_dir> <output_dir> ")
    exit(1)


inputDir = sys.argv[1]
outputDir = sys.argv[2]

for imageName in os.listdir(inputDir):
    if imageName[-5]!='0':
        continue
    path1 = os.path.join(inputDir,imageName)
    path2 = path1.replace("img0","img1")
    im1 = cv2.imread(path1)
    im2 = cv2.imread(path2)

    if im1 is None:
        print("ERROR: can not read image\t",path1)
        continue
    if im2 is None:
        print("ERROR: can not read image\t",path2)
        continue

    im1 = ResizeSingle(im1,desired_single_h,desired_single_w)
    im2 = ResizeSingle(im2,desired_single_h,desired_single_w)
    im_concat = cv2.hconcat([im1, im2])
    im_concat = cv2.resize(im_concat, (desired_concat_size, desired_concat_size))
    outPath = os.path.join(outputDir,imageName.replace("-img0",""))
    print("Writing\t",imageName.replace("-img0",""))
    cv2.imwrite(outPath,im_concat)






