"""This is to test the color transfer between two COCO images."""
from color_transfer import color_transfer
import numpy as np
import scipy
import scipy.misc

from pycocotools.coco import COCO

ann_path = '/data/public/MSCOCO/annotations/instances_train2017.json'
root = '/data/public/MSCOCO/train2017'


def initialize():
    coco = COCO(ann_path)
    imgIds = coco.getImgIds(catIds=[], imgIds=[])
    dataset_size = len(imgIds)
    return coco, imgIds, dataset_size


def transfer(dataset_size, coco, imgIds, root):
    # Select a random image
    while True:
        current_id = np.random.randint(dataset_size)
        image_info = coco.loadImgs(imgIds[current_id % dataset_size])[0]
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
        break

    # Select another image as color transfer source.
    new_id = np.random.randint(dataset_size) + current_id - 1
    image_info = coco.loadImgs(imgIds[new_id % dataset_size])[0]
    image_url = image_info['coco_url']
    image_url_split = image_url.split('/')
    image_path = '{}/{}'.format(root, image_url_split[-1])

    target_image = scipy.misc.imread(image_path, mode='RGB')
    transfer_image = color_transfer(target_image, source_image)

    return source_image, target_image, transfer_image


if __name__ == '__main__':
    coco, imgIds, dataset_size = initialize()

    np.random.seed(0)
    for i in range(10):
        source_image, target_image, transfer_image = \
            transfer(dataset_size, coco, imgIds, root)
        scipy.misc.imsave(
            'color_transfer_res/source_image_{}.png'.format(i),
            source_image)
        scipy.misc.imsave(
            'color_transfer_res/target_image_{}.png'.format(i),
            target_image)
        scipy.misc.imsave(
            'color_transfer_res/transfer_image_{}.png'.format(i),
            transfer_image)
