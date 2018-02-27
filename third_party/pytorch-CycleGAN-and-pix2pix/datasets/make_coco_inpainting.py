"""This is the file to create inpainting dataset for training the pix2pix."""
import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image inpainting pairs for training pix2pix.')
parser.add_argument('--fold', dest='fold', help='input folder for inpainting', type=str,
                    default="/data/public/MSCOCO/train2017")
parser.add_argument('--des_fold', dest='des_fold', help='output folder for inpainting', type=str,
                    default="/data/guided_inpainting/dataset/coco_inpainting")
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))


img_list = os.listdir(args.fold)
num_imgs = min(args.num_imgs, len(img_list))

if not os.path.isdir(args.des_fold):
    os.makedirs(args.des_fold)

for n in range(num_imgs):
    name_A = img_list[n]
    path_A = os.path.join(args.fold, name_A)
    if args.use_AB:
        name_B = name_A.replace('_A.', '_B.')
    else:
        name_B = name_A
    path_B = path_A
    if os.path.isfile(path_A) and os.path.isfile(path_B):
        name_AB = name_A
        path_AB = os.path.join(args.des_fold, name_AB)
        im_A = cv2.imread(path_A, cv2.CV_LOAD_IMAGE_COLOR)
        im_B = cv2.imread(path_B, cv2.CV_LOAD_IMAGE_COLOR)
        im_B[70: 186, 70:186, 0] = 117 - 0.5
        im_AB = np.concatenate([im_A, im_B], 1)
        cv2.imwrite(path_AB, im_AB)
