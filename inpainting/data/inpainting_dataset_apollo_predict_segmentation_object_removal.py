"""Inpainting dataset for pix2pix HD. This is for guided inpainting. The
experiment is that we crop an object, put it in another image and paste it
back (including some background of another image). Different from
inpainting_dataset_guided which only inpaints one class, we use all the
images from COCO here. """
from copy import deepcopy
import numpy as np
import scipy
import scipy.misc
import torch
import os

from inpainting.data.base_dataset import BaseDataset
from inpainting.data.image_folder import make_dataset

np.random.seed(0)

class InpaintingDatasetApolloPredictSegmentationObjectRemoval(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.paths, self.annpaths = [],[]

        imagelistfile = open("{}/road01_ins_train.lst".format(self.root))
        for line in imagelistfile:
            self.paths.append(self.root + '/' + line.strip().split('\t')[0])
            self.annpaths.append(self.root + '/' + line.strip().split('\t')[1])
        imagelistfile.close()
        imagelistfile = open("{}/road02_ins_train.lst".format(self.root))
        for line in imagelistfile:
            self.paths.append(self.root + '/' + line.strip().split('\t')[0])
            self.annpaths.append(self.root + '/' + line.strip().split('\t')[1])
        imagelistfile.close()
        imagelistfile = open("{}/road03_ins_train.lst".format(self.root))
        for line in imagelistfile:
            self.paths.append(self.root + '/' + line.strip().split('\t')[0])
            self.annpaths.append(self.root + '/' + line.strip().split('\t')[1])
        imagelistfile.close()

        # if opt.isTrain:
        #     imagelistfile = open("{}/road01_ins_train.lst".format(self.root))
        #     for line in imagelistfile:
        #         self.paths.append(self.root + '/' + line.strip().split('\t')[0])
        #         self.annpaths.append(self.root + '/' + line.strip().split('\t')[1])
        #     imagelistfile.close()
        #     imagelistfile = open("{}/road02_ins_train.lst".format(self.root))
        #     for line in imagelistfile:
        #         self.paths.append(self.root + '/' + line.strip().split('\t')[0])
        #         self.annpaths.append(self.root + '/' + line.strip().split('\t')[1])
        #     imagelistfile.close()
        #     imagelistfile = open("{}/road03_ins_train.lst".format(self.root))
        #     for line in imagelistfile:
        #         self.paths.append(self.root + '/' + line.strip().split('\t')[0])
        #         self.annpaths.append(self.root + '/' + line.strip().split('\t')[1])
        #     imagelistfile.close()
        # else:
        #     imagelistfile = open("{}/road01_ins_test.lst".format(self.root))
        #     for line in imagelistfile:
        #         self.paths.append(self.root + '/' + line.strip().split('\t')[0])
        #         self.annpaths.append(self.root + '/' + line.strip().split('\t')[1])
        #     imagelistfile.close()
        #     imagelistfile = open("{}/road02_ins_test.lst".format(self.root))
        #     for line in imagelistfile:
        #         self.paths.append(self.root + '/' + line.strip().split('\t')[0])
        #         self.annpaths.append(self.root + '/' + line.strip().split('\t')[1])
        #     imagelistfile.close()
        #     imagelistfile = open("{}/road03_ins_test.lst".format(self.root))
        #     for line in imagelistfile:
        #         self.paths.append(self.root + '/' + line.strip().split('\t')[0])
        #         self.annpaths.append(self.root + '/' + line.strip().split('\t')[1])
        #     imagelistfile.close()

        self.label_nc = opt.label_nc

        self.dataset_size = len(self.paths)

    def __getitem__(self, index):
        """We take an image from COCO, find a random object in the image and
        then remove the object background in the bounding box. This is used
        as input for training. The output is the original image. """
        current_id = index
        while True:
            image_path = self.paths[current_id % self.dataset_size]
            image_annpath = self.annpaths[current_id % self.dataset_size]
            image = scipy.misc.imread(image_path, mode='RGB')
            # image_ann = scipy.misc.imread(image_annpath, mode='I')
            image_seg = scipy.misc.imread(image_annpath, mode='L')
            image_height, image_width, _ = image.shape
            # image_seg = np.zeros((self.seg_nc, image_height, image_width))
            # if self.seg_nc == 35:
            #     for i in range(-1,34):
            #         image_seg[i+1, image_ann==i] = 1
            # else:
            #     mapping = [0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,5,6,6,7,7,7,7,7,7,7,7,7]
            #     for i in range(-1,34):
            #         image_seg[mapping[i], image_ann==i] = 1
            # if self.label_nc == 8:
            #     mapping = [0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,5,6,6,7,7,7,7,7,7,7,7,7]
            #     for i in range(-1,34):
            #         image_seg[image_seg==i] = mapping[i]


            
            if image_height > 128 and image_width > 128:
                break
            current_id = current_id + 1


        # define mask
        mask = np.zeros((image_height, image_width))
        moving_ids = [33,161,34,162,35,163,36,164,37,165,38,166,39,167,40,168]
        for moving_id in moving_ids:
            mask[image_seg == moving_id] = 1



        # mask = np.zeros((image_height, image_width))
        # hole_y = np.random.randint(image_height - 32)
        # hole_x = np.random.randint(image_width - 32)
        # hole_height = np.random.randint(low=32, high=min(int(image_height/2), image_height-hole_y+1))   # [low, high)
        # hole_width = np.random.randint(low=32, high=min(int(image_width/2), image_width-hole_x+1))
        # mask[hole_y: hole_y+hole_height, hole_x:hole_x+hole_width] = 1

        if self.label_nc == 8:
            mapping = {
            0: 0,
            1: 0,
            17: 5,
            33: 7,
            161: 7,
            34: 7,
            162: 7,
            35: 7,
            163: 7,
            36: 6,
            164: 6,
            37: 6,
            165: 6,
            38: 7,
            166: 7,
            39: 7,
            167: 7,
            40: 7,
            168: 7,
            49: 1,
            50: 1,
            65: 3,
            66: 3,
            67: 2,
            81: 3,
            82: 3,
            83: 3,
            84: 2,
            85: 3,
            86: 3,
            97: 2,
            98: 2,
            99: 2,
            100: 2,
            113: 4,
            255: 0
            }
            # mapping = [0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,5,6,6,7,7,7,7,7,7,7,7,7]
            # for i in range(-1,34):
            #     image_seg[image_seg==i] = mapping[i]
            image_seg_reduced = image_seg
            for (key, value) in mapping.items():
                image_seg_reduced[image_seg==key] = value
            image_seg = image_seg_reduced



        # resize image
        image_resized = scipy.misc.imresize(image, [self.opt.fineSize, self.opt.fineSize])  # fineSize x fineSize x 3  # noqa 501
        image_resized = np.rollaxis(image_resized, 2, 0)  # 3 x fineSize x fineSize  # noqa 501

        mask_resized = scipy.misc.imresize(mask, [self.opt.fineSize,
                                                  self.opt.fineSize])
        mask_resized[mask_resized > 0] = 1  # fineSize x fineSize

        if self.opt.isTrain:
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

        # mask_resized = np.tile(mask_resized, (3, 1, 1))

        # resize segmentation
        image_seg_resized = scipy.misc.imresize(image_seg, [self.opt.fineSize, self.opt.fineSize], interp='nearest', mode='F')
        
        # image_seg_resized = np.zeros((self.seg_nc, self.opt.fineSize, self.opt.fineSize))
        # for i in range(self.seg_nc):
        #     image_seg_resized[i] = scipy.misc.imresize(image_seg[i], [self.opt.fineSize, self.opt.fineSize])

        # normalize
        image_resized = image_resized / 122.5 - 1

        input_image = np.copy(image_resized)
        input_image[np.tile(mask_resized, (3, 1, 1)) == 1] = 0



        input_image_seg = np.copy(image_seg_resized)
        input_image_seg[mask_resized == 1] = 0
        # mask_resized = np.tile(mask_resized, (3, 1, 1))


        # change from numpy to pytorch tensor
        mask_resized = torch.from_numpy(mask_resized).float()
        mask_resized = mask_resized.unsqueeze(0)

        input_image = torch.from_numpy(input_image).float()
        input_image_seg = torch.from_numpy(input_image_seg).float()
        input_image_seg = input_image_seg.unsqueeze(0)
        image_resized = torch.from_numpy(image_resized).float()
        image_seg_resized = torch.from_numpy(image_seg_resized).float()
        image_seg_resized = image_seg_resized.unsqueeze(0)

        if self.opt.use_pretrained_model:
            input_image_unsqueezed = input_image_seg.unsqueeze(0)
            mask_resized_unsqueezed = mask_resized.unsqueeze(0)
            generated = self.pretrained_model.inference(input_image_unsqueezed, mask_resized_unsqueezed)
            input_image_l2 = generated.data[0]

            input_dict = {'input': input_image_l2, 'mask': mask_resized,
                          'image': image_seg_resized,
                          'original_image': image_resized,
                          'masked_image': input_image,
                          'path': image_path,
                          'input_original': input_image_seg}
        else:
            input_dict = {'input': input_image_seg, 'mask': mask_resized,
                          'image': image_seg_resized,
                          'original_image': image_resized,
                          'masked_image': input_image,
                          'path': image_path}

        return input_dict

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'InpaintingDatasetApolloPredictSegmentationObjectRemoval'
