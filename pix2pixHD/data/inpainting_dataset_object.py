"""Inpainting dataset for pix2pix HD. This is for guided inpainting. The first experiment is that we crop a center
part, do style transfer and then paste it back. """
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
        self.catIds = self.coco.getCatIds(catNms=['bus'])
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)
        self.dataset_size = len(self.imgIds)

    def __getitem__(self, index):
        """We use image with hole as the input label."""
        image_info = self.coco.loadImgs(self.imgIds[index])[0]
        image_url = image_info['coco_url']
        image_url_split = image_url.split('/')
        image_path = '{}/{}'.format(self.root, image_url_split[-1])

        image = scipy.misc.imread(image_path, mode='RGB')
        annIds = self.coco.getAnnIds(imgIds=image_info['id'], catIds=self.catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        mask = self.coco.annToMask(anns[0])

        image2 = scipy.misc.imresize(image, [self.opt.fineSize, self.opt.fineSize])
        if self.opt.use_color is True:
            gray_image = np.tile(color.rgb2gray(image2), (3, 1, 1))
        else:
            gray_image = 110/255.0
        image2 = np.rollaxis(image2, 2, 0)

        mask_image2 = scipy.misc.imresize(mask, [self.opt.fineSize, self.opt.fineSize])
        mask_image2[mask_image2 > 0] = 1
        mask_image2 = np.tile(mask_image2, (3, 1, 1))
        mask_image3 = 1 - mask_image2

        # gray image
        image3 = image2 * mask_image3 + gray_image * mask_image2 * 255

        image3 = image3/122.5-1
        image2 = image2/122.5-1

        image_with_hole = torch.from_numpy(image3).float()
        image_pytorch = torch.from_numpy(image2).float()
        mask_pytorch = torch.from_numpy(mask_image2).float()

        inst_tensor = feat_tensor = 0

        input_dict = {'label': image_with_hole, 'inst': mask_pytorch, 'image': image_pytorch,
                      'feat': feat_tensor, 'path': image_path}

        return input_dict

    def __len__(self):
        return len(self.imgIds)

    def name(self):
        return 'InpaintingDatasetObject'
