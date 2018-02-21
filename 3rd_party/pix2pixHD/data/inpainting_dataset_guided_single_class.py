"""Inpainting dataset for pix2pix HD. This is for guided inpainting. The experiment is that we crop an object,
put it in another image and paste it back (including some background of another image). This is an initial experiment
for a single class (bus). """
import numpy as np
from PIL import Image
import scipy
import torch

from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from pycocotools.coco import COCO


class InpaintingDatasetGuided(BaseDataset):
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

        # resize image
        image_resized = scipy.misc.imresize(image, [self.opt.fineSize, self.opt.fineSize])  # fineSize x fineSize x 3
        image_resized = np.rollaxis(image_resized, 2, 0)  # 3 x fineSize x fineSize

        mask_resized = scipy.misc.imresize(mask, [self.opt.fineSize, self.opt.fineSize])
        mask_resized[mask_resized > 0] = 1  # fineSize x fineSize
        a = np.where(mask_resized != 0)
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        mask_resized = np.tile(mask_resized, (3, 1, 1))

        # get another image as the guide image
        image_info = self.coco.loadImgs(self.imgIds[(index+1) % self.dataset_size])[0]
        image_url = image_info['coco_url']
        image_url_split = image_url.split('/')
        guided_path = '{}/{}'.format(self.root, image_url_split[-1])
        guided_image = scipy.misc.imread(guided_path, mode='RGB')

        guided_image_resized = scipy.misc.imresize(guided_image, [self.opt.fineSize, self.opt.fineSize])
        guided_image_resized = np.rollaxis(guided_image_resized, 2, 0)

        # Place the object in a random place of the guided image.
        object_x = bbox[2]
        object_y = bbox[0]
        object_height = bbox[1] - object_y + 1
        object_width = bbox[3] - object_x + 1

        place_x = np.random.randint(self.opt.fineSize - object_width+1)
        place_y = np.random.randint(self.opt.fineSize - object_height+1)

        guided_image_cropped = guided_image_resized[:, place_y:place_y + object_height, place_x:place_x + object_width]
        mask_image_cropped = mask_resized[:, object_y:object_y + object_height, object_x:object_x + object_width]
        object_image = image_resized[:, object_y:object_y + object_height, object_x:object_x + object_width]
        guided_object = object_image * mask_image_cropped + guided_image_cropped * (1 - mask_image_cropped)

        input_image = np.copy(image_resized)
        input_image[:, object_y:object_y + object_height, object_x:object_x + object_width] = guided_object

        feat_tensor = 0

        input_image = input_image/122.5 - 1
        image_resized = image_resized/122.5 - 1
        input_image = torch.from_numpy(input_image).float()
        image_resized = torch.from_numpy(image_resized).float()
        mask_resized_tmp = np.copy(mask_resized)
        mask_resized[:,  object_y:object_y + object_height, object_x:object_x + object_width] = 1
        mask_resized[mask_resized_tmp == 1] = 0
        input_image = np.copy(image_resized)
        input_image[mask_resized == 1] = 0
        mask_resized = torch.from_numpy(mask_resized).float()

        input_dict = {'label': input_image, 'inst': mask_resized, 'image': image_resized,
                      'feat': feat_tensor, 'path': image_path}

        return input_dict

    def __len__(self):
        return len(self.imgIds)

    def name(self):
        return 'InpaintingDatasetGuided'
