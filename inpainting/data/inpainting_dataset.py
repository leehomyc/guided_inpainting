"""Inpainting dataset for pix2pix HD."""
import os.path
from inpainting.data.base_dataset import BaseDataset, get_params, get_transform, normalize
from inpainting.data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import random

class InpaintingDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.paths = sorted(make_dataset(self.root))

        self.dataset_size = len(self.paths)

    def __getitem__(self, index):
        """We use image with hole as the input label."""
        path = self.paths[index]
        image = Image.open(path)
        params = get_params(self.opt, image.size)

        transform_image = get_transform(self.opt, params)
        image = transform_image(image.convert('RGB'))

        image_with_hole = image.clone()

        if self.opt.random_location:
            hole_x_begin = random.randint(0, self.opt.fineSize / 2 + 2 * self.opt.overlapPred)
            hole_y_begin = random.randint(0, self.opt.fineSize / 2 + 2 * self.opt.overlapPred)
            hole_x_end = int(hole_x_begin + self.opt.fineSize / 2 - 2 * self.opt.overlapPred)
            hole_y_end = int(hole_y_begin + self.opt.fineSize / 2 - 2 * self.opt.overlapPred)

        else:
            hole_y_begin = int(self.opt.fineSize / 4 + self.opt.overlapPred)
            hole_y_end = int(self.opt.fineSize / 2 + self.opt.fineSize / 4 - self.opt.overlapPred)
            hole_x_begin = hole_y_begin
            hole_x_end = hole_y_end

        image_with_hole[0, hole_y_begin:hole_y_end, hole_x_begin: hole_x_end] = 2 * 117.0 / 255.0 - 1.0
        image_with_hole[1, hole_y_begin:hole_y_end, hole_x_begin: hole_x_end] = 2 * 104.0 / 255.0 - 1.0
        image_with_hole[2, hole_y_begin:hole_y_end, hole_x_begin: hole_x_end] = 2 * 123.0 / 255.0 - 1.0

        mask = np.zeros((3, self.opt.fineSize, self.opt.fineSize))
        mask[:, hole_y_begin:hole_y_end, hole_x_begin: hole_x_end] = 1 


        inst_tensor = feat_tensor = 0

        # change from numpy to pytorch tensor
        mask = torch.from_numpy(mask).float()

        input_dict = {'input': image_with_hole, 'mask':mask, 'inst': inst_tensor, 'image': image,
                      'feat': feat_tensor, 'path': path}

        return input_dict

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'InpaintingDataset'
