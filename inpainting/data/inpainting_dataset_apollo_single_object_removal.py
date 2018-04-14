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

from inpainting.data.base_dataset import BaseDataset
from inpainting.data.image_folder import make_dataset

class InpaintingDatasetApolloSingleObjectRemoval(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # self.paths = sorted(make_dataset(self.root))
        self.paths, self.annpaths, self.inspaths = [],[],[]
        imagelistfile = open("{}/road01_ins_train.lst".format(self.root))
        for line in imagelistfile:
            self.paths.append(self.root + '/' + line.strip().split('\t')[0])
            self.annpaths.append(self.root + '/' + line.strip().split('\t')[1])
            self.inspaths.append(self.root + '/' + line.strip().split('\t')[1].replace('_bin','').split('.png')[0] + '_instanceIds.png')
        imagelistfile.close()
        imagelistfile = open("{}/road02_ins_train.lst".format(self.root))
        for line in imagelistfile:
            self.paths.append(self.root + '/' + line.strip().split('\t')[0])
            self.annpaths.append(self.root + '/' + line.strip().split('\t')[1])
            self.inspaths.append(self.root + '/' + line.strip().split('\t')[1].replace('_bin','').split('.png')[0] + '_instanceIds.png')
        imagelistfile.close()
        imagelistfile = open("{}/road03_ins_train.lst".format(self.root))
        for line in imagelistfile:
            self.paths.append(self.root + '/' + line.strip().split('\t')[0])
            self.annpaths.append(self.root + '/' + line.strip().split('\t')[1])
            self.inspaths.append(self.root + '/' + line.strip().split('\t')[1].replace('_bin','').split('.png')[0] + '_instanceIds.png')
        imagelistfile.close()

        self.dataset_size = len(self.paths)

    def __getitem__(self, index):
        """We take an image from COCO, find a random object in the image and
        then remove the object background in the bounding box. This is used
        as input for training. The output is the original image. """
        current_id = index
        while True:
            image_path = self.paths[current_id % self.dataset_size]
            image_annpath = self.annpaths[current_id % self.dataset_size]
            image_inspath = self.inspaths[current_id % self.dataset_size]
            image = scipy.misc.imread(image_path, mode='RGB')
            image_ann = scipy.misc.imread(image_annpath, mode='L')
            image_ins = scipy.misc.imread(image_inspath, mode='I')
            image_height, image_width, _ = image.shape

            # define mask
            mask = np.zeros((image_height, image_width))
            moving_ids = [33,161,34,162,35,163,36,164,37,165,38,166,39,167,40,168]
            movingins_ids = []
            for moving_id in moving_ids:
                instance_id = moving_id * 1000
                # print(np.any(image_ins==instance_id), instance_id)
                while np.any(image_ins==instance_id):
                    movingins_ids.append(instance_id)
                    instance_id = instance_id + 1
            # print('test {}'.format(len(movingins_ids)))
            if len(movingins_ids)>0:
                break

            current_id = current_id + 1


        ins_id = movingins_ids[np.random.randint(low = 0, high = len(movingins_ids))]
        # print(ins_id)
        mask[image_ins == ins_id] = 1

        # resize image
        image_resized = scipy.misc.imresize(image, [self.opt.fineSize, self.opt.fineSize])  # fineSize x fineSize x 3  # noqa 501
        image_resized = np.rollaxis(image_resized, 2, 0)  # 3 x fineSize x fineSize  # noqa 501

        mask_resized = scipy.misc.imresize(mask, [self.opt.fineSize,
                                                  self.opt.fineSize])
        mask_resized[mask_resized > 0] = 1  # fineSize x fineSize
        a = np.where(mask_resized != 0)
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        mask_resized = np.tile(mask_resized, (3, 1, 1))

        # find the bounding box of the image.
        object_x = bbox[2]
        object_y = bbox[0]
        object_height = bbox[1] - object_y + 1
        object_width = bbox[3] - object_x + 1

        # normalize
        image_resized = image_resized / 122.5 - 1

        # only keep the object foreground and remove the object background.
        # mask_resized_tmp = np.copy(mask_resized)
        mask_resized[:, object_y:object_y + object_height, object_x:object_x + object_width] = 1  # noqa 501
        # mask_resized[mask_resized_tmp == 1] = 0
        input_image = np.copy(image_resized)
        input_image[mask_resized == 1] = 0

        # change from numpy to pytorch tensor
        mask_resized = torch.from_numpy(mask_resized).float()
        input_image = torch.from_numpy(input_image).float()
        image_resized = torch.from_numpy(image_resized).float()

        if self.opt.use_pretrained_model:
            input_image_unsqueezed = input_image.unsqueeze(0)
            mask_resized_unsqueezed = mask_resized.unsqueeze(0)
            generated = self.pretrained_model.inference(input_image_unsqueezed, mask_resized_unsqueezed)
            input_image_l2 = generated.data[0]

            input_dict = {'input': input_image_l2, 'mask': mask_resized,
                          'image': image_resized,
                          'path': image_path,
                          'input_original': input_image}
        else:
            input_dict = {'input': input_image, 'mask': mask_resized,
                          'image': image_resized,
                          'path': image_path}

        return input_dict

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'InpaintingDatasetApolloSingleObjectRemoval'
