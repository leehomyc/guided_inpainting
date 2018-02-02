"""Inpainting dataset for pix2pix HD. This is for image harmonization. We only do image harmonization on the object
of the image. """
import numpy as np
from PIL import Image
import scipy
import torch
from torch.autograd import Variable
import torch.nn as nn

import data.AdaptiveInstanceNormalization as Adain
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from data.models import decoder, vgg_normalised
from pycocotools.coco import COCO
import skimage.io as io
from skimage import color


class InpaintingDatasetObject(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.annFile = opt.ann_path
        self.coco = COCO(self.annFile)
        self.imgIds = self.coco.getImgIds(catIds=[], imgIds=[])
        self.dataset_size = len(self.imgIds)

    def __getitem__(self, index):
        # Select an image with object.
        current_id = index
        while True:
            image_info = self.coco.loadImgs(self.imgIds[current_id % self.dataset_size])[0]
            image_url = image_info['coco_url']
            image_url_split = image_url.split('/')
            image_path = '{}/{}'.format(self.root, image_url_split[-1])

            image = scipy.misc.imread(image_path, mode='RGB')
            annIds = self.coco.getAnnIds(imgIds=image_info['id'], areaRng=[100, float('inf')], iscrowd=None)
            if len(annIds) == 0:
                # This image has no annotations. We have to switch to the next image.
                current_id = current_id + 1
                continue
            anns = self.coco.loadAnns(annIds)
            mask = self.coco.annToMask(anns[np.random.randint(0, len(anns))])
            break

        # Resize the image to 256x256x3
        image_resized = scipy.misc.imresize(image, [self.opt.fineSize, self.opt.fineSize])
        # Get a gray scale image. Note that the range of gray image is [0,1]
        gray_image = np.tile(color.rgb2gray(image_resized), (3, 1, 1))
        # change the shape to 3x256x256
        image_resized = np.rollaxis(image_resized, 2, 0)

        # Resize the object mask
        mask_resized = scipy.misc.imresize(mask, [self.opt.fineSize, self.opt.fineSize])
        mask_resized[mask_resized > 0] = 1
        # Change the mask size to 3x256x256
        mask_resized = np.tile(mask_resized, (3, 1, 1))

        # Composite the gray image and the color background
        image_composite = image_resized * (1 - mask_resized) + gray_image * mask_resized * 255

        # Normalize
        image_composite = image_composite / 122.5 - 1
        image_resized = image_resized / 122.5 - 1

        # Change to PyTorch CUDA Tensor.
        image_composite = torch.from_numpy(image_composite).float()
        image_resized = torch.from_numpy(image_resized).float()
        mask_resized = torch.from_numpy(mask_resized).float()

        input_dict = {'label': image_composite, 'inst': mask_resized, 'image': image_resized,
                      'feat': 0, 'path': image_path}

        return input_dict

    def __len__(self):
        return len(self.imgIds)

    def name(self):
        return 'InpaintingDatasetObject'
