"""This is to test the color transfer between two COCO images."""
from color_transfer import color_transfer
import numpy as np
import os
import scipy
import scipy.misc

from pycocotools.coco import COCO

ann_path = '/data/public/MSCOCO/annotations/instances_train2017.json'
root = '/data/public/MSCOCO/train2017'

res_folder = '../../guided_inpainting_output/color_transfer_obj_res'

os.makedirs(res_folder, exist_ok=True)


def initialize():
    coco = COCO(ann_path)
    imgIds = coco.getImgIds(catIds=[], imgIds=[])
    dataset_size = len(imgIds)
    return coco, imgIds, dataset_size


# noinspection PyShadowingNames
def transfer(dataset_size, coco, img_ids, root):
    # Select a random image
    current_id = np.random.randint(dataset_size)
    while True:
        image_info = coco.loadImgs(img_ids[current_id % dataset_size])[0]
        image_url = image_info['coco_url']
        image_url_split = image_url.split('/')
        image_path = '{}/{}'.format(root, image_url_split[-1])

        source_image = scipy.misc.imread(image_path, mode='RGB')
        annIds = coco.getAnnIds(imgIds=image_info['id'],
                                areaRng=[1000, float('inf')], iscrowd=None)
        if len(annIds) == 0:
            # This image has no annotations. We have to switch to the next
            # image.
            current_id = np.random.randint(dataset_size)
            continue

        anns = coco.loadAnns(annIds)
        mask = coco.annToMask(anns[0])

        ########################################
        #  remove backgrounds of source image.##
        ########################################

        # change mask size to 3 X H X W
        mask = np.tile(mask, (3, 1, 1))
        # change size to H x W x 3.
        mask = np.rollaxis(mask, 0, 3)
        source_image[mask == 0] = 0
        break

    # Select another image as color transfer source.
    new_id = np.random.randint(dataset_size) + current_id - 1
    image_info = coco.loadImgs(img_ids[new_id % dataset_size])[0]
    image_url = image_info['coco_url']
    image_url_split = image_url.split('/')
    image_path = '{}/{}'.format(root, image_url_split[-1])

    target_image = scipy.misc.imread(image_path, mode='RGB')

    # Source image provides the content, and the target image provides color.
    transfer_image = color_transfer(target_image, source_image)
    transfer_image[mask == 0] = 0
    return source_image, target_image, transfer_image


if __name__ == '__main__':
    coco, imgIds, dataset_size = initialize()

    np.random.seed(0)
    for i in range(10):
        source_image, target_image, transfer_image = \
            transfer(dataset_size, coco, imgIds, root)
        scipy.misc.imsave(
            '{}/source_image_{}.png'.format(res_folder, i),
            source_image)
        scipy.misc.imsave(
            '{}/target_image_{}.png'.format(res_folder, i),
            target_image)
        scipy.misc.imsave(
            '{}/transfer_image_{}.png'.format(res_folder, i),
            transfer_image)
