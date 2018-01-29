"""Inpainting dataset for pix2pix HD. This is for guided inpainting. The first experiment is that we crop a center
part, do style transfer and then paste it back. """
from PIL import Image
import torch
import torch.nn as nn

import data.AdaptiveInstanceNormalization as Adain
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from data.models import decoder, vgg_normalised
from torch.autograd import Variable


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


class InpaintingDatasetGuided(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.paths = sorted(make_dataset(self.root))
        self.dataset_size = len(self.paths)
        self.vgg = load_vgg_model()
        self.decoder = load_decoder_model()
        self.content_weight = opt.content_weight

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
        path = self.paths[index]
        image = Image.open(path)

        # Load another image as the guidance image.
        style_path = self.paths[(index + 1) % self.dataset_size]
        style_image = Image.open(style_path)

        params = get_params(self.opt, image.size)

        transform_image = get_transform(self.opt, params)
        image = transform_image(image.convert('RGB'))  # The image's range is 0-1, and the shape is 3xHxW. The type
        # is Torch Float Tensor.
        style_image = transform_image(style_image.convert('RGB'))

        # Do style transfer
        content_image = image.unsqueeze(0)  # Change from 3D to 4D
        style_image = style_image.unsqueeze(0)
        # image = image.float()
        content_image = Variable(content_image, volatile=True)
        style_image = Variable(style_image, volatile=True)
        image_style_transferred = self.style_transfer(content_image, style_image, self.content_weight)
        # print(image_style_transferred.shape)

        image_hole_style_transferred = image.clone()

        hole_y_begin = int(self.opt.fineSize / 4 + self.opt.overlapPred)
        hole_y_end = int(self.opt.fineSize / 2 + self.opt.fineSize / 4 - self.opt.overlapPred)
        hole_x_begin = hole_y_begin
        hole_x_end = hole_y_end

        image_hole_style_transferred[:, hole_y_begin:hole_y_end, hole_x_begin: hole_x_end] = \
            image_style_transferred[:, :, hole_y_begin:hole_y_end, hole_x_begin: hole_x_end]

        inst_tensor = feat_tensor = 0

        input_dict = {'label': image_hole_style_transferred, 'inst': inst_tensor, 'image': image,
                      'feat': feat_tensor, 'path': path}

        return input_dict

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'InpaintingDataset'
