"""Harmonization data loader. """
import cv2
import numpy as np
import scipy
import scipy.misc
import torch

from harmonization.data.base_dataset import BaseDataset
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
    :param source_mask: the image as the masks of the source. Value 0 means the
        pixel is the background and should be ignored.
    :return:
    """
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype('float32')
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    # Reverse the values of the mask so that value 1 means it should be ignored.
    source_mask = 1 - source_mask

    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(
        source, source_mask)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(
        target)

    # The target image is gray scale image. We do not apply color transfer.
    if lStdTar == 0 or aStdTar == 0 or bStdTar == 0:
        return source

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
class InpaintingDatasetHarm(BaseDataset):
    def initialize(self, opt):
        np.random.seed(
            0)  # make sure each time the training input is identical.  # noqa 501

        self.opt = opt
        self.root = opt.dataroot
        self.annFile = opt.ann_path
        self.coco = COCO(self.annFile)
        self.imgIds = self.coco.getImgIds(catIds=[], imgIds=[])
        self.dataset_size = len(self.imgIds)

    def retrieve_current_image(self, index):
        """ Retrieve the current image.

        :param index: an integer as the index of the image.
        :return: image, mask and the image path.
        """
        while True:
            image_info = self.coco.loadImgs(
                self.imgIds[index % self.dataset_size])[0]
            image_url = image_info['coco_url']
            image_url_split = image_url.split('/')
            image_path = '{}/{}'.format(self.root, image_url_split[-1])

            source_image = scipy.misc.imread(image_path, mode='RGB')
            annIds = self.coco.getAnnIds(imgIds=image_info['id'],
                                         areaRng=[self.opt.range_threshold,
                                                  float('inf')],  # noqa 501
                                         iscrowd=None)
            if len(annIds) == 0:
                # This image has no suitable annotations. We have to switch to the next image.  # noqa
                index = index + 1
                continue
            break
        anns = self.coco.loadAnns(annIds)
        mask_hw = self.coco.annToMask(anns[np.random.randint(0, len(
            anns))])  # find a random object
        mask_hw[mask_hw > 0] = 1
        return source_image, mask_hw, image_path

    def color_transfer(self, image, mask, index):
        """ Color transfer for current image.

        :param image: source image.
        :param mask: a single channel mask.
        :param index: index of the source image.
        :return:
        """
        source_image = image.copy()
        mask_hw = mask.copy()
        # change mask size to 3 X H X W
        mask_chw = np.tile(mask_hw, (3, 1, 1))
        # change size to H x W x 3.
        mask_hwc = np.rollaxis(mask_chw, 0, 3)
        source_image[mask_hwc == 0] = 0

        # Select another image as source of color.
        target_id = np.random.randint(self.dataset_size) + index - 1
        target_image_info = \
            self.coco.loadImgs(self.imgIds[target_id % self.dataset_size])[0]
        target_image_url = target_image_info['coco_url']
        target_image_url_split = target_image_url.split('/')
        target_image_path = \
            '{}/{}'.format(self.root, target_image_url_split[-1])
        target_image = scipy.misc.imread(target_image_path, mode='RGB')

        # Source image provides the content, and the target image provides
        # color.
        transfer_image = color_transfer(source_image, target_image,
                                        mask_hw)
        transfer_image[mask_hwc == 0] = image[mask_hwc == 0]
        return transfer_image

    def __getitem__(self, index):
        """We take an image from COCO, find a random object in the image and
        then use another image to transfer the color and paste it back to the
        original image. """
        original_image, mask_hw, image_path = self.retrieve_current_image(index)

        # color transfer using another image.
        input_image = self.color_transfer(original_image, mask_hw, index)

        input_image = \
            scipy.misc.imresize(input_image, [self.opt.fineSize, self.opt.fineSize])  # fineSize x fineSize x 3  # noqa 501
        input_image_chw = np.rollaxis(input_image, 2, 0)  # 3 x fineSize x fineSize  # noqa 501
        input_image_chw = input_image_chw / 122.5 - 1  # normalize

        original_image = \
            scipy.misc.imresize(original_image, [self.opt.fineSize, self.opt.fineSize])  # fineSize x fineSize x 3  # noqa 501
        original_image_chw = np.rollaxis(original_image, 2, 0)  # 3 x fineSize x fineSize  # noqa 501
        original_image_chw = original_image_chw / 122.5 - 1  # normalize

        mask_resized = scipy.misc.imresize(
            mask_hw, [self.opt.fineSize, self.opt.fineSize])
        mask_resized[mask_resized > 0] = 1  # fineSize x fineSize
        mask_resized_chw = np.tile(mask_resized, (3, 1, 1))
        
        # change from numpy to pytorch tensor
        mask_resized_chw = torch.from_numpy(mask_resized_chw).float()
        input_image_chw = torch.from_numpy(input_image_chw).float()
        original_image_chw = torch.from_numpy(original_image_chw).float()

        input_dict = {'input': input_image_chw, 'mask': mask_resized_chw,
                      'image': original_image_chw,
                      'path': image_path}

        return input_dict

    def __len__(self):
        return len(self.imgIds)

    def name(self):
        return 'InpaintingDatasetHarm'
