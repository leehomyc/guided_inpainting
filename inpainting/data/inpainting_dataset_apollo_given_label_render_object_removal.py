"""Inpainting dataset for pix2pix HD. This is for guided inpainting. The
experiment is that we crop an object, put it in another image and paste it
back (including some background of another image). Different from
inpainting_dataset_guided which only inpaints one class, we use all the
images from COCO here. """
from copy import deepcopy
import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
import torch
import os

from inpainting.data.base_dataset import BaseDataset
from inpainting.data.image_folder import make_dataset

np.random.seed(0)

class InpaintingDatasetApolloGivenLabelRenderObjectRemoval(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.paths, self.annpaths, self.renderpaths = [],[],[]
        if opt.isTrain:
            imagelistfile = open("{}/split/train.txt".format(self.root))
            for line in imagelistfile:
                self.paths.append(self.root + '/image/' + line.strip() + '.jpg')
                self.annpaths.append(self.root + '/label/' + line.strip() + '.png')
                self.renderpaths.append(self.root + '/label_render/' + line.strip() + '.png')
            imagelistfile.close()
        else:
            imagelistfile = open("{}/split/train.txt".format(self.root))
            for line in imagelistfile:
                self.paths.append(self.root + '/image/' + line.strip() + '.jpg')
                self.annpaths.append(self.root + '/label/' + line.strip() + '.png')
                self.renderpaths.append(self.root + '/label_render/' + line.strip() + '.png')
            imagelistfile.close()


        self.seg_nc = opt.seg_nc

        self.dataset_size = len(self.paths)

    def __getitem__(self, index):
        """We take an image from COCO, find a random object in the image and
        then remove the object background in the bounding box. This is used
        as input for training. The output is the original image. """
        current_id = index
        while True:
            image_path = self.paths[current_id % self.dataset_size]
            image_annpath = self.annpaths[current_id % self.dataset_size]
            image_renderpath = self.renderpaths[current_id % self.dataset_size]
            image = scipy.misc.imread(image_path, mode='RGB')
            # image_ann = scipy.misc.imread(image_annpath, mode='I')
            image_seg = scipy.misc.imread(image_annpath, mode='I')
            image_render = scipy.misc.imread(image_renderpath)
            image_height, image_width, _ = image.shape
            # image_seg = np.zeros((self.seg_nc, image_height, image_width))
            # if self.seg_nc == 35:
            #     for i in range(-1,34):
            #         image_seg[i+1, image_ann==i] = 1
            # else:
            #     mapping = [0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,5,6,6,7,7,7,7,7,7,7,7,7]
            #     for i in range(-1,34):
            #         image_seg[mapping[i], image_ann==i] = 1
            # if self.seg_nc == 8:
            #     mapping = [0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,5,6,6,7,7,7,7,7,7,7,7,7]
            #     for i in range(-1,34):
            #         image_seg[image_seg==i] = mapping[i]


            
            if image_height > 128 and image_width > 128:
                break
            current_id = current_id + 1

        ## define mask
        #mask = np.zeros((image_height, image_width))
        #moving_ids = [33,161,34,162,35,163,36,164,37,165,38,166,39,167,40,168]
        #for moving_id in moving_ids:
        #    mask[image_seg == moving_id] = 1

        # resize image
        image_resized = scipy.misc.imresize(image, [self.opt.fineSize, self.opt.fineSize])  # fineSize x fineSize x 3  # noqa 501
        image_resized = np.rollaxis(image_resized, 2, 0)  # 3 x fineSize x fineSize  # noqa 501

        #mask_resized = scipy.misc.imresize(mask, [self.opt.fineSize,
        #                                          self.opt.fineSize])
        #mask_resized[mask_resized > 0] = 1  # fineSize x fineSize

        # define mask
        mask = np.zeros((self.opt.fineSize, self.opt.fineSize))
        moving_ids = [33,161,34,162,35,163,36,164,37,165,38,166,39,167,40,168]
        for moving_id in moving_ids:
            mask[image_seg == moving_id] = 1
        mask_resized = mask

        # # random move the mask
        padding_size = int(self.opt.fineSize/4)
        mask_padding = np.zeros((self.opt.fineSize + 2*padding_size, self.opt.fineSize + 2*padding_size))
        random_x = np.random.randint(low = int(padding_size/3), high = int(padding_size)) * (1 if np.random.randn()>0 else -1) + padding_size
        random_y = np.random.randint(low = int(padding_size/3), high = int(padding_size)) * (1 if np.random.randn()>0 else -1) + padding_size
        mask_padding[random_x:random_x+self.opt.fineSize, random_y:random_y+self.opt.fineSize] = mask_resized
        mask_resized = mask_padding[padding_size:(self.opt.fineSize+padding_size), padding_size:(self.opt.fineSize+padding_size)]

        # padding_size = int(self.opt.fineSize/10)
        # mask_padding = np.zeros((self.opt.fineSize + 2*padding_size, self.opt.fineSize + 2*padding_size))
        # random_x = np.random.randint(low = int(padding_size/2), high = 2*padding_size)
        # random_y = np.random.randint(low = int(padding_size/2), high = 2*padding_size)
        # mask_padding[random_x:random_x+self.opt.fineSize, random_y:random_y+self.opt.fineSize] = mask_resized
        # mask_resized = mask_padding[0:self.opt.fineSize, 0:self.opt.fineSize]

        if self.opt.mask_dilation_iter > 0:
            struct1 = scipy.ndimage.generate_binary_structure(2,2)
            mask_resized = scipy.ndimage.binary_dilation(mask_resized, structure=struct1, iterations=self.opt.mask_dilation_iter).astype(mask_resized.dtype)

        mask_resized = np.tile(mask_resized, (3, 1, 1))

        # # resize segmentation
        # image_seg_resized = scipy.misc.imresize(image_seg, [self.opt.fineSize, self.opt.fineSize], interp='nearest', mode='F')
        # # image_seg_resized = np.zeros((self.seg_nc, self.opt.fineSize, self.opt.fineSize))
        # # for i in range(self.seg_nc):
        # #     image_seg_resized[i] = scipy.misc.imresize(image_seg[i], [self.opt.fineSize, self.opt.fineSize])

        # if self.seg_nc == 8:
        #     mapping = {
        #     0: 0,
        #     1: 0,
        #     17: 5,
        #     33: 7,
        #     161: 7,
        #     34: 7,
        #     162: 7,
        #     35: 7,
        #     163: 7,
        #     36: 6,
        #     164: 6,
        #     37: 6,
        #     165: 6,
        #     38: 7,
        #     166: 7,
        #     39: 7,
        #     167: 7,
        #     40: 7,
        #     168: 7,
        #     49: 1,
        #     50: 1,
        #     65: 3,
        #     66: 3,
        #     67: 2,
        #     81: 3,
        #     82: 3,
        #     83: 3,
        #     84: 2,
        #     85: 3,
        #     86: 3,
        #     97: 2,
        #     98: 2,
        #     99: 2,
        #     100: 2,
        #     113: 4,
        #     255: 0
        #     }
        #     # mapping = [0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,5,6,6,7,7,7,7,7,7,7,7,7]
        #     # for i in range(-1,34):
        #     #     image_seg[image_seg==i] = mapping[i]
        #     image_seg_resized_reduced = image_seg_resized
        #     for (key, value) in mapping.items():
        #         image_seg_resized_reduced[image_seg_resized==key] = value
        #     image_seg_resized = image_seg_resized_reduced

        # resize render
        image_render_resized = scipy.misc.imresize(image_render, [self.opt.fineSize, self.opt.fineSize], interp='nearest', mode='F')

        if self.seg_nc == 8:
            mapping_render = {
            0: 0,
            1: 5,
            2: 7,
            32: 7,
            3: 7,
            4: 6,
            5: 6,
            33: 7,
            34: 7,
            35: 7,
            7: 1,
            8: 1,
            14: 3,
            15: 3,
            16: 2,
            19: 3,
            20: 3,
            21: 3,
            17: 2,
            18: 3,
            22: 3,
            25: 2,
            31: 4
            }
            # mapping = [0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,5,6,6,7,7,7,7,7,7,7,7,7]
            # for i in range(-1,34):
            #     image_seg[image_seg==i] = mapping[i]
            image_render_resized_reduced = image_render_resized
            for (key, value) in mapping_render.items():
                image_render_resized_reduced[image_render_resized==key] = value
            image_render_resized = image_render_resized_reduced
        image_seg_resized = image_render_resized

        # normalize
        image_resized = image_resized / 122.5 - 1

        input_image = np.copy(image_resized)
        input_image[mask_resized == 1] = 0

        # change from numpy to pytorch tensor
        mask_resized = torch.from_numpy(mask_resized).float()
        input_image = torch.from_numpy(input_image).float()
        image_resized = torch.from_numpy(image_resized).float()
        image_seg_resized = torch.from_numpy(image_seg_resized).float()
        image_seg_resized = image_seg_resized.unsqueeze(0)

        if self.opt.use_pretrained_model:
            input_image_unsqueezed = input_image.unsqueeze(0)
            mask_resized_unsqueezed = mask_resized.unsqueeze(0)
            generated = self.pretrained_model.inference(input_image_unsqueezed, mask_resized_unsqueezed)
            input_image_l2 = generated.data[0]

            input_dict = {'input': input_image_l2, 'mask': mask_resized,
                          'image': image_resized,
                          'input_seg': image_seg_resized,
                          'path': image_path,
                          'input_original': input_image}
        else:
            input_dict = {'input': input_image, 'mask': mask_resized,
                          'image': image_resized,
                          'input_seg': image_seg_resized,
                          'path': image_path}

        return input_dict

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'InpaintingDatasetApolloGivenLabelRenderObjectRemoval'