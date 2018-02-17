"""Inpainting dataset for pix2pix HD. This is for guided inpainting. The
experiment is that we crop an object, put it in another image and paste it
back (including some background of another image). Different from
inpainting_dataset_guided which only inpaints one class, we use all the
images from COCO here. """
import cv2
import numpy as np
import scipy
import scipy.misc
import torch

from data.base_dataset import BaseDataset
from pycocotools.coco import COCO


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


# noinspection PyAttributeOutsideInit
class InpaintingDatasetGuided(BaseDataset):
    def initialize(self, opt):
        np.random.seed(
            0)  # make sure each time the training input is identical.  # noqa 501

        self.opt = opt
        self.root = opt.dataroot
        self.annFile = opt.ann_path
        self.coco = COCO(self.annFile)
        self.imgIds = self.coco.getImgIds(catIds=[], imgIds=[])
        self.dataset_size = len(self.imgIds)

    def __getitem__(self, index):
        """We take an image from COCO, find a random object in the image and
        then use another image to transfer the color and paste it back to the
        original image. """
        current_id = index
        while True:
            image_info = self.coco.loadImgs(
                self.imgIds[current_id % self.dataset_size])[0]
            image_url = image_info['coco_url']
            image_url_split = image_url.split('/')
            image_path = '{}/{}'.format(self.root, image_url_split[-1])

            source_image = scipy.misc.imread(image_path, mode='RGB')
            annIds = self.coco.getAnnIds(imgIds=image_info['id'],
                                         areaRng=[1000, float('inf')],
                                         # The area of the object must be greater than 1000  # noqa 501
                                         iscrowd=None)
            if len(annIds) == 0:
                # This image has no annotations. We have to switch to the
                # next image.
                current_id = current_id + 1
                continue
            break

        anns = self.coco.loadAnns(annIds)
        mask = self.coco.annToMask(anns[np.random.randint(0, len(
            anns))])  # find a random object  # noqa 501

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

        # Select another image as source of color.
        new_id = np.random.randint(self.dataset_size) + current_id - 1
        image_info = \
            self.coco.loadImgs(self.img_ids[new_id % self.dataset_size])[0]
        image_url = image_info['coco_url']
        image_url_split = image_url.split('/')
        image_path = '{}/{}'.format(self.root, image_url_split[-1])
        target_image = scipy.misc.imread(image_path, mode='RGB')

        # Source image provides the content, and the target image provides
        # color.
        transfer_image = color_transfer(source_image, target_image,
                                        mask_single_channel)

        # Put the transfer image back to the original image.
        source_image[mask == 1] = transfer_image[mask == 1]

        # resize image
        image_resized = scipy.misc.imresize(source_image, [self.opt.fineSize,
                                                    self.opt.fineSize])  # fineSize x fineSize x 3  # noqa 501
        image_resized = np.rollaxis(image_resized, 2,
                                    0)  # 3 x fineSize x fineSize  # noqa 501

        mask_resized = scipy.misc.imresize(mask, [self.opt.fineSize,
                                                  self.opt.fineSize])
        mask_resized[mask_resized > 0] = 1  # fineSize x fineSize
        a = np.where(mask_resized != 0)
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        mask_resized = np.tile(mask_resized, (3, 1, 1))

        # find the bounding box of the image.
        object_x = bbox[2]
        object_y = bbox[0]
        object_height = bbox[1] - object_y + 1
        object_width = bbox[3] - object_x + 1

        # normalize
        image_resized = image_resized / 122.5 - 1

        # only keep the object foreground and remove the object background.
        mask_resized_tmp = np.copy(mask_resized)
        mask_resized[:, object_y:object_y + object_height,
        object_x:object_x + object_width] = 1  # noqa 501
        mask_resized[mask_resized_tmp == 1] = 0
        input_image = np.copy(image_resized)
        input_image[mask_resized == 1] = 0

        # change from numpy to pytorch tensor
        mask_resized = torch.from_numpy(mask_resized).float()
        input_image = torch.from_numpy(input_image).float()
        image_resized = torch.from_numpy(image_resized).float()

        input_dict = {'input': input_image, 'mask': mask_resized,
                      'image': image_resized,
                      'path': image_path}

        return input_dict

    def __len__(self):
        return len(self.imgIds)

    def name(self):
        return 'InpaintingDatasetGuided'
