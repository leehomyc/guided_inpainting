"""Inpainting dataset for pix2pix HD. This is for guided inpainting. The
experiment is that we crop an object, put it in another image and paste it
back (including some background of another image). Different from
inpainting_dataset_guided which only inpaints one class, we use all the
images from COCO here. """
import numpy as np
import scipy
import scipy.misc
import torch

from data.base_dataset import BaseDataset
from pycocotools.coco import COCO


class InpaintingDatasetGuided(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.annFile = opt.ann_path
        self.coco = COCO(self.annFile)
        self.imgIds = self.coco.getImgIds(catIds=[], imgIds=[])
        self.dataset_size = len(self.imgIds)

    def __getitem__(self, index):
        """We take an image from COCO, find a random object in the image and
        then remove the object background in the bounding box. This is used
        as input for training. The output is the original image. """
        current_id = index
        while True:
            image_info = self.coco.loadImgs(
                self.imgIds[current_id % self.dataset_size])[0]
            image_url = image_info['coco_url']
            image_url_split = image_url.split('/')
            image_path = '{}/{}'.format(self.root, image_url_split[-1])

            image = scipy.misc.imread(image_path, mode='RGB')
            annIds = self.coco.getAnnIds(imgIds=image_info['id'],
                                         areaRng=[100, float('inf')],  # The area of the object must be greater than 100  # noqa 501
                                         iscrowd=None)
            if len(annIds) == 0:
                # This image has no annotations. We have to switch to the
                # next image.
                current_id = current_id + 1
                continue
            anns = self.coco.loadAnns(annIds)
            mask = self.coco.annToMask(anns[np.random.randint(0, len(anns))])  # find a random object  # noqa 501
            break

        # resize image
        image_resized = scipy.misc.imresize(image, [self.opt.fineSize, self.opt.fineSize])  # fineSize x fineSize x 3  # noqa 501
        image_resized = np.rollaxis(image_resized, 2, 0)  # 3 x fineSize x fineSize  # noqa 501

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
        mask_resized[:, object_y:object_y + object_height, object_x:object_x + object_width] = 1  # noqa 501
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
