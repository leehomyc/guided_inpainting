"""This is to test the color transfer between two COCO images."""
from color_transfer import color_transfer
import cv2
import numpy as np
import os
import scipy
import scipy.misc

from pycocotools.coco import COCO

ann_path = '/data/public/MSCOCO/annotations/instances_train2017.json'
root = '/data/public/MSCOCO/train2017'

res_folder = '../../guided_inpainting_output/color_transfer_obj_nb_res'

os.makedirs(res_folder, exist_ok=True)


def image_stats(image, mask=None):
    # compute the mean and standard deviation of each channel
    # note when mask==1, the pixel should be ignored.
    (l, a, b) = cv2.split(image)
    if mask is not None:
        mask = mask.astype(bool)
    l = np.ma.array(l, mask=mask)
    a = np.ma.array(a, mask=mask)
    b = np.ma.array(b, mask=mask)

    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return lMean, lStd, aMean, aStd, bMean, bStd


def color_transfer(source, target, source_mask):
    """

    :param source: the image as the contents.
    :param target: the image as the colors.
    :param source_mask: the image as the masks of the source
    :return:
    """
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype('float32')
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(
        source, source_mask)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(
        target)
    if lStdTar == 0 or aStdTar == 0 or bStdTar == 0:
        return source
    print(image_stats(target))

    # subtract the means from the target image
    (l, a, b) = cv2.split(source)
    l -= lMeanSrc
    a -= aMeanSrc
    b -= bMeanSrc

    # scale by the standard deviations
    l = (lStdSrc / lStdTar) * l
    a = (aStdSrc / aStdTar) * a
    b = (bStdSrc / bStdTar) * b

    # add in the source mean
    l += lMeanTar
    a += aMeanTar
    b += bMeanTar

    # clip the pixel intensities to [0, 255] if they fall outside
    # this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    # return the color transferred image
    return transfer


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
        mask_single_channel = mask.copy()
        # The object pixel is 1 and the background is 0
        mask_single_channel[mask_single_channel > 0] = 1
        # The object pixel is 0 and the background is 1
        mask_single_channel = 1 - mask_single_channel
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
    transfer_image = color_transfer(source_image, target_image,
                                    mask_single_channel)

    transfer_image[mask == 0] = 0
    return source_image, target_image, transfer_image


if __name__ == '__main__':
    coco, imgIds, dataset_size = initialize()

    np.random.seed(0)
    for i in range(100):
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
