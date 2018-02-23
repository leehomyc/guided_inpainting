"""Inpainting grid dataset for pix2pix HD. We would like to try splitting the images
into nine grids and use eight grids as input with the hole being the output."""
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torch


# noinspection PyArgumentList
class InpaintingGridDataset(BaseDataset):
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
        image = transform_image(image.convert('RGB'))  # image is transformed into torch.FloatTensor of shape
        # channel x height x idth.

        image_height = image_width = self.opt.fineSize
        grid_height = grid_width = int(self.opt.fineSize / 3)
        image_grid = torch.FloatTensor(24, grid_height, grid_width)
        grid_id = 0
        for i in range(0, image_height, grid_height):
            for j in range(0, image_width, grid_width):
                if i == grid_height and j == grid_width:
                    continue
                image_grid[grid_id*3:grid_id*3+3, :, :] = image[:, i:i+grid_height, j:j+grid_width]
                grid_id = grid_id + 1
        missing_grid = image[:, grid_height:grid_height*2, grid_width:grid_width*2]

        inst_tensor = feat_tensor = 0

        input_dict = {'label': image_grid, 'inst': inst_tensor, 'image': missing_grid,
                      'feat': feat_tensor, 'path': path, 'original_image': image}

        return input_dict

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'InpaintingDataset'
