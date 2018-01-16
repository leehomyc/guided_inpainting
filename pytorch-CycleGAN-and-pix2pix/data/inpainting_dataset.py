import os.path
import random
import torchvision.transforms as transforms
import torch
from .base_dataset import BaseDataset
from .image_folder import make_dataset
from PIL import Image


class InpaintingDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.paths = sorted(make_dataset(self.root))

        assert (opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert('RGB')
        image = image.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        image = self.transform(image)

        w = image.size(2)
        h = image.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        image = image[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        input_nc = self.opt.input_nc

        # Flip the image for data augmentation
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(image.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            image = image.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = image[0, ...] * 0.299 + image[1, ...] * 0.587 + image[2, ...] * 0.114
            image = tmp.unsqueeze(0)

        image_with_hole = image.clone()

        hole_y_begin = int(self.opt.fineSize/4 + self.opt.overlapPred)
        hole_y_end = int(self.opt.fineSize/2 + self.opt.fineSize/4-self.opt.overlapPred)
        hole_x_begin = hole_y_begin
        hole_x_end = hole_y_end

        image_with_hole[0, hole_y_begin:hole_y_end, hole_x_begin: hole_x_end] = 2 * 117.0 / 255.0-1.0
        image_with_hole[1, hole_y_begin:hole_y_end, hole_x_begin: hole_x_end] = 2 * 104.0 / 255.0 - 1.0
        image_with_hole[2, hole_y_begin:hole_y_end, hole_x_begin: hole_x_end] = 2 * 123.0 / 255.0 - 1.0

        return {'A': image_with_hole, 'B': image,
                'A_paths': self.root, 'B_paths': self.root}

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'AlignedDataset'
