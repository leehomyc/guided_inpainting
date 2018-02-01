"""Inpainting dataset for pix2pix HD. This is for guided inpainting. The experiment is that we crop an object,
put it in another image and paste it back (including some background of another image). """
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


def load_decoder_model():
    """Load the decoder model which is converted from the Torch lua model using
    git@github.com:clcarwin/convert_torch_to_pytorch.git.

    :return: The decoder model as described in the paper.
    """
    this_decoder = decoder.decoder
    this_decoder.load_state_dict(torch.load('data/models/decoder.pth'))
    this_decoder.eval()
    return this_decoder


def load_vgg_model():
    """Load the VGG model."""
    this_vgg = vgg_normalised.vgg_normalised
    this_vgg.load_state_dict(torch.load('data/models/vgg_normalised.pth'))
    """
    This is to ensure that the vgg is the same as the model used in PyTorch lua as below:
    vgg = torch.load(opt.vgg)
    for i=53,32,-1 do
        vgg:remove(i)
    end
    This actually removes 22 layers from the VGG model.
    """
    this_vgg = nn.Sequential(*list(this_vgg)[:-22])
    this_vgg.eval()
    return this_vgg


class InpaintingDatasetGuidedStyleTransfer(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.annFile = opt.ann_path
        self.coco = COCO(self.annFile)
        self.catIds = self.coco.getCatIds(catNms=['bus'])
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)
        self.dataset_size = len(self.imgIds)

    def style_transfer(self, content_image, style_img, content_weight=0.25):
        """Style transfer between content image and style image."""
        style_feature = self.vgg(style_img)  # torch.Size([1, 512, 16, 16])
        content_feature = self.vgg(content_image)  # torch.Size([1, 512, 16, 16])
        input = torch.cat((content_feature, style_feature), 0)
        adain = Adain.AdaptiveInstanceNormalization()
        target_feature = adain(input)
        target_feature = (1 - content_weight) * target_feature + content_weight * content_feature
        return self.decoder(target_feature).data

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

        # get bounding box
        # mask_image2 = np.tile(mask_image2, (3, 1, 1))
        a = np.where(mask_resized != 0)
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])

        # get another image as the guide image
        guided_path = self.paths[(index + 1) % self.dataset_size]
        guided_image = Image.open(guided_path)

        guided_image_resized = scipy.misc.imresize(guided_image, [self.opt.fineSize, self.opt.fineSize])
        guided_image_resized = np.rollaxis(guided_image_resized, 2, 0)

        # Place the object in a random place of the guided image.
        object_x = bbox[2]
        object_y = bbox[0]
        object_height = bbox[1] - object_y + 1
        object_width = bbox[3] - object_x + 1

        place_x = np.random.randomint(self.opt.fineSize - object_width+1)
        place_y = np.random.randomint(self.opt.fineSize - object_height+1)

        guided_image_cropped = guided_image_resized[:, place_y:place_y + object_height, place_x:place_x + object_width]
        mask_image_cropped = mask_resized[:, place_y:place_y + object_height, place_x:place_x + object_width]
        object_image = image_resized[:, place_y:place_y + object_height, place_x:place_x + object_width]
        guided_object = object_image * mask_image_cropped + guided_image_cropped * (1 - mask_image_cropped)

        input_image = guided_image_resized.clone()
        input_image[:, place_y:place_y + object_height, place_x:place_x + object_width] = guided_object

        inst_tensor = feat_tensor = 0

        input_dict = {'label': input_image, 'inst': inst_tensor, 'image': image_resized,
                      'feat': feat_tensor, 'path': image_path}

        return input_dict

    def __len__(self):
        return len(self.imgIds)

    def name(self):
        return 'InpaintingDatasetGuided'
