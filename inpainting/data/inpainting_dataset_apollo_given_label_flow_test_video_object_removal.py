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
from os import listdir
from os.path import isfile, join

from inpainting.data.base_dataset import BaseDataset
from inpainting.data.image_folder import make_dataset

np.random.seed(0)

class InpaintingDatasetApolloGivenLabelFlowTestVideoObjectRemoval(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        files = ['_'.join(f.split('_')[0:4]) for f in listdir(self.root) if isfile(join(self.root, f))]

        filehash = set(files)
        fileid = [i for i in filehash]

        self.paths = [self.root+'/'+i+'_gt.jpg' for i in fileid]
        self.renderpaths = [self.root+'/'+i+'_render.png' for i in fileid]
        self.maskpaths = [self.root+'/'+i+'_mask.png' for i in fileid]
        self.warppaths = [self.root+'/'+i+'_warp.jpg' for i in fileid]

        self.seg_nc = opt.seg_nc

        self.dataset_size = len(self.paths)

    def __getitem__(self, index):
        """We take an image from COCO, find a random object in the image and
        then remove the object background in the bounding box. This is used
        as input for training. The output is the original image. """
        current_id = index
        while True:
            image_path = self.paths[current_id % self.dataset_size]
            image_renderpath = self.renderpaths[current_id % self.dataset_size]
            image_maskpath = self.maskpaths[current_id % self.dataset_size]
            image_warppath = self.warppaths[current_id % self.dataset_size]

            image = scipy.misc.imread(image_path, mode='RGB')
            image_render = scipy.misc.imread(image_renderpath)
            image_mask = scipy.misc.imread(image_maskpath, mode='L')
            image_warp = scipy.misc.imread(image_warppath, mode='RGB')
            image_height, image_width, _ = image.shape

            # downsample the image by factor 2

            # image = scipy.misc.imresize(image, [int(image_height/4), int(image_width/4)]) 
            # image_warp = scipy.misc.imresize(image_warp, [int(image_height/4), int(image_width/4)]) 
            # image_render = scipy.misc.imresize(image_render, [int(image_height/4), int(image_width/4)], interp='nearest', mode='F')
            # image_mask = scipy.misc.imresize(image_mask, [int(image_height/4), int(image_width/4)], interp='nearest', mode='F')
            # image_height, image_width, _ = image.shape

            # resize to a reasonable size (pytorch needs to keep the size which could be divided by 4)
            image = scipy.misc.imresize(image, [688, 848]) 
            image_warp = scipy.misc.imresize(image_warp, [688, 848]) 
            image_render = scipy.misc.imresize(image_render, [688, 848], interp='nearest', mode='F')
            image_mask = scipy.misc.imresize(image_mask, [688, 848], interp='nearest', mode='F')
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


        # crop images
        image_resized = image.copy()
        image_resized = np.rollaxis(image_resized, 2, 0)  # 3 x fineSize x fineSize 

        warp_resized = image_warp.copy()
        warp_resized = np.rollaxis(warp_resized, 2, 0)  # 3 x fineSize x fineSize

        mask_resized = image_mask.copy()
        # scale mask to [0,1]
        mask_resized = mask_resized/255

        #dilation
        struct1 = scipy.ndimage.generate_binary_structure(2,2)
        mask_resized = scipy.ndimage.binary_dilation(mask_resized, structure=struct1, iterations=3).astype(mask_resized.dtype)


        mask_resized = np.tile(mask_resized, (3, 1, 1))

        image_render_resized = image_render.copy()
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
            31: 4,
            255: 0
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
        warp_resized = warp_resized / 122.5 - 1

        input_image = np.copy(image_resized)
        input_image[mask_resized > 0] = 0

        

        # change from numpy to pytorch tensor
        mask_resized = torch.from_numpy(mask_resized).float()
        input_image = torch.from_numpy(input_image).float()
        image_resized = torch.from_numpy(image_resized).float()
        warp_resized = torch.from_numpy(warp_resized).float()
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
                          'conditional_image': warp_resized,
                          'path': image_path,
                          'input_original': input_image}
        else:
            input_dict = {'input': input_image, 'mask': mask_resized,
                          'image': image_resized,
                          'input_seg': image_seg_resized,
                          'conditional_image': warp_resized,
                          'path': image_path}

        return input_dict

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'InpaintingDatasetApolloGivenLabelFlowTestVideoObjectRemoval'
