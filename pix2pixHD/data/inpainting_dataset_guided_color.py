"""Inpainting dataset for pix2pix HD. This is for guided inpainting. The first experiment is that we crop a center
part, do style transfer and then paste it back. """
import numpy as np
from PIL import Image
import torch

from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset


class InpaintingDatasetColor(BaseDataset):
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
        original_image = transform_image(image.convert('RGB'))  # The image's range is -1-1, and the shape is 3xHxW. The type
        # is Torch Float Tensor.
        gray_image = transform_image(image.convert('L').convert('RGB'))

        hole_y_begin = int(self.opt.fineSize / 4 + self.opt.overlapPred)
        hole_y_end = int(self.opt.fineSize / 2 + self.opt.fineSize / 4 - self.opt.overlapPred)
        hole_x_begin = hole_y_begin
        hole_x_end = hole_y_end

        image_with_hole = original_image.clone()

        image_with_hole[:, hole_y_begin:hole_y_end, hole_x_begin: hole_x_end] = \
            gray_image[:, hole_y_begin:hole_y_end, hole_x_begin: hole_x_end]

        inst_tensor = feat_tensor = 0

        if self.opt.use_mask is True:
            mask = np.ones((1, self.opt.fineSize, self.opt.fineSize))
            mask[:, hole_y_begin:hole_y_end, hole_x_begin: hole_x_end] = 0
            mask = torch.from_numpy(mask)
            mask = mask.float()
        else:
            mask = 0

        input_dict = {'label': image_with_hole, 'inst': inst_tensor, 'image': original_image,
                      'feat': feat_tensor, 'path': path, 'mask': mask}

        return input_dict

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'InpaintingDataset'
