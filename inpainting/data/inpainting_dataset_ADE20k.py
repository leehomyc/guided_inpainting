"""Inpainting dataset for pix2pix HD. This is for guided inpainting. The
experiment is that we crop an object, put it in another image and paste it
back (including some background of another image). Different from
inpainting_dataset_guided which only inpaints one class, we use all the
images from COCO here. """
import numpy as np
import os
import scipy
import scipy.io as sio
import scipy.misc
import torch

from inpainting.data.base_dataset import BaseDataset


class InpaintingDatasetADE20k(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.index_ade20k = sio.loadmat('{}/ADE20K_2016_07_26/index_ade20k.mat'.format(self.root))
        self.dataset_size = len(self.index_ade20k['index']['folder'][0][0][0])

    def __getitem__(self, index):
        """We take an image from COCO, find a random object in the image and
        then remove the object background in the bounding box. This is used
        as input for training. The output is the original image. """
        current_id = index
        while True:
            folder = self.index_ade20k['index']['folder'][0][0][0][current_id % self.dataset_size][0]
            filename = self.index_ade20k['index']['filename'][0][0][0][current_id % self.dataset_size][0]
            image_path = os.path.join(self.root, folder, filename)
            image = scipy.misc.imread(image_path, mode='RGB')
            image_height, image_width, _ = image.shape
            if image_height > 128 and image_width > 128:
                break
            current_id = current_id + 1

        basename = image_path.split('.')[0]
        seg_name = '{}_seg.png'.format(basename)
        seg = scipy.misc.imread(seg_name, mode='RGB')

        mask = np.zeros((image_height, image_width))
        hole_y = np.random.randint(image_height - 32)
        hole_x = np.random.randint(image_width - 32)
        hole_height = np.random.randint(low=32, high=min(int(image_height / 2),
                                                         image_height - hole_y + 1))  # [low, high)
        hole_width = np.random.randint(low=32, high=min(int(image_width / 2),
                                                        image_width - hole_x + 1))
        mask[hole_y: hole_y + hole_height, hole_x:hole_x + hole_width] = 1

        # resize image
        image_resized = scipy.misc.imresize(image, [self.opt.fineSize,
                                                    self.opt.fineSize])  # fineSize x fineSize x 3  # noqa 501
        image_resized = np.rollaxis(image_resized, 2,
                                    0)  # 3 x fineSize x fineSize  # noqa 501
        seg_resized = scipy.misc.imresize(seg, [self.opt.fineSize,
                                                self.opt.fineSize])
        seg_resized = np.rollaxis(seg_resized, 2, 0)

        mask_resized = scipy.misc.imresize(mask, [self.opt.fineSize,
                                                  self.opt.fineSize])
        mask_resized[mask_resized > 0] = 1  # fineSize x fineSize
        mask_resized = np.tile(mask_resized, (3, 1, 1))

        # normalize
        image_resized = image_resized / 122.5 - 1
        seg_resized = seg_resized / 122.5 - 1

        input_image = np.copy(image_resized)
        input_image[mask_resized == 1] = 0

        # change from numpy to pytorch tensor
        mask_resized = torch.from_numpy(mask_resized).float()
        input_image = torch.from_numpy(input_image).float()
        image_resized = torch.from_numpy(image_resized).float()
        input_seg = 0
        if self.opt.use_seg is True:
            input_seg = torch.from_numpy(seg_resized).float()

        input_dict = {'input': input_image, 'mask': mask_resized,
                      'image': image_resized,
                      'path': image_path,
                      'input_seg': input_seg}

        return input_dict

    def __len__(self):
        return len(self.index_ade20k['index']['folder'][0][0][0])

    def name(self):
        return 'InpaintingDatasetADEk'
