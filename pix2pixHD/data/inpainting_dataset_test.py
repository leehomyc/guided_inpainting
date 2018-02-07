"""Dataset for guided inpainting testing. """
import numpy as np
import scipy
import torch

from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from pycocotools.coco import COCO

from skimage import color

IMG_PAIRS = [{
    'object_img_id': 5037,
    'background_img_id': 447342,
    'object_id': 0,
    'object_composite_x': 5,
    'object_composite_y': 127,
    'object_composite_width': 115,
    'object_composite_height': 122,
}
]


class InpaintingDatasetTest(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.annFile = opt.ann_path
        self.coco = COCO(self.annFile)
        self.dataset_size = len(IMG_PAIRS)

    def __getitem__(self, index):
        """We take an image from COCO, find a random object in the image and then remove the object background in the
        bounding box. This is used as input for training. The output is the original image. """
        # load object image and mask
        image_info = self.coco.loadImgs(ids=IMG_PAIRS[index]['object_img_id'])[0]
        image_url = image_info['coco_url']
        image_url_split = image_url.split('/')
        image_path = '{}/{}'.format(self.root, image_url_split[-1])

        image = scipy.misc.imread(image_path, mode='RGB')
        object_image_height, object_image_width, _ = image.shape
        annIds = self.coco.getAnnIds(imgIds=image_info['id'], iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        mask = self.coco.annToMask(anns[IMG_PAIRS[index]['object_id']])

        # load background image
        image_info = self.coco.loadImgs(ids=IMG_PAIRS[index]['background_img_id'])[0]
        image_url = image_info['coco_url']
        image_url_split = image_url.split('/')
        background_image_path = '{}/{}'.format(self.root, image_url_split[-1])
        image_background = scipy.misc.imread(background_image_path, mode='RGB')

        # find bounding box
        mask[mask > 0] = 1  # fineSize x fineSize
        a = np.where(mask != 0)
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])

        object_x = bbox[2]
        object_y = bbox[0]
        object_height = bbox[1] - object_y + 1
        object_width = bbox[3] - object_x + 1

        # compute the new size of the image
        object_image_resize_height = int(
            object_image_height * IMG_PAIRS[index]['object_composite_height'] / object_height)
        object_image_resize_width = int(
            object_image_width * IMG_PAIRS[index]['object_composite_width'] / object_width)

        if self.opt.model == 'inpainting_test':
            # resize object image
            image_resized = scipy.misc.imresize(image, [object_image_resize_height,
                                                        object_image_resize_width])  # fineSize x fineSize x 3
            image_resized = np.rollaxis(image_resized, 2, 0)  # 3 x fineSize x fineSize

            # resize object mask
            mask_resized = scipy.misc.imresize(mask, [object_image_resize_height, object_image_resize_width])
            mask_resized[mask_resized > 0] = 1  # fineSize x fineSize
            a = np.where(mask_resized != 0)
            bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
            mask_resized = np.tile(mask_resized, (3, 1, 1))

            # get bounding box
            object_x = bbox[2]
            object_y = bbox[0]
            object_height = IMG_PAIRS[index]['object_composite_height']
            object_width = IMG_PAIRS[index]['object_composite_width']

            # normalize object image
            image_resized = image_resized / 122.5 - 1

            # get the sub-image that contains the object
            object_image = image_resized[:, object_y:object_y + object_height, object_x:object_x + object_width]
            object_mask = mask_resized[:, object_y:object_y + object_height, object_x:object_x + object_width]
            object_image[object_mask == 0] = 0

            # resize and normalize the background image
            image_background_resized = scipy.misc.imresize(image_background, [256, 256])  # fineSize x fineSize x 3
            image_background_resized = np.rollaxis(image_background_resized, 2, 0)
            image_background_resized = image_background_resized / 122.5 - 1

            # image composition
            new_object_x = IMG_PAIRS[index]['object_composite_x']
            new_object_y = IMG_PAIRS[index]['object_composite_y']
            image_composite = np.copy(image_background_resized)
            image_composite[:, new_object_y: new_object_y+object_height, new_object_x: new_object_x+object_width] = object_image

            mask_composite = np.zeros(image_composite.shape)
            mask_composite[:, new_object_y:new_object_y + object_height,
            new_object_x:new_object_x + object_width] = 1 - object_mask
        elif self.opt.model == 'harmonization_test':
            # resize object image
            image_resized = scipy.misc.imresize(image, [object_image_resize_height,
                                                        object_image_resize_width])  # fineSize x fineSize x 3
            image_resized = np.tile(color.rgb2gray(image_resized), (3, 1, 1))  # change to gray
            # image_resized = np.rollaxis(image_resized, 2, 0)  # 3 x fineSize x fineSize

            # resize object mask
            mask_resized = scipy.misc.imresize(mask, [object_image_resize_height, object_image_resize_width])
            mask_resized[mask_resized > 0] = 1  # fineSize x fineSize
            a = np.where(mask_resized != 0)
            bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
            mask_resized = np.tile(mask_resized, (3, 1, 1))

            # get bounding box
            object_x = bbox[2]
            object_y = bbox[0]
            object_height = IMG_PAIRS[index]['object_composite_height']
            object_width = IMG_PAIRS[index]['object_composite_width']

            # normalize object image from [0,1] to [-1, 1]
            image_resized = image_resized * 2 - 1

            # get the sub-image that contains the object
            object_image = image_resized[:, object_y:object_y + object_height, object_x:object_x + object_width]
            object_mask = mask_resized[:, object_y:object_y + object_height, object_x:object_x + object_width]
            # object_image[object_mask == 0] = 0

            # resize and normalize the background image
            image_background_resized = scipy.misc.imresize(image_background, [256, 256])  # fineSize x fineSize x 3
            image_background_resized = np.rollaxis(image_background_resized, 2, 0)
            image_background_resized = image_background_resized / 122.5 - 1

            # image composition
            new_object_x = IMG_PAIRS[index]['object_composite_x']
            new_object_y = IMG_PAIRS[index]['object_composite_y']
            image_composite = np.copy(image_background_resized)

            image_background_resized_crop = image_background_resized[:, new_object_y: new_object_y + object_height,
            new_object_x: new_object_x + object_width]
            image_background_resized_crop_composite = image_background_resized_crop * (1-object_mask) + object_image * object_mask
            image_composite[:, new_object_y: new_object_y + object_height,
            new_object_x: new_object_x + object_width] = image_background_resized_crop_composite

            mask_composite = np.zeros(image_composite.shape)
            mask_composite[:, new_object_y:new_object_y + object_height,
            new_object_x:new_object_x + object_width] = object_mask
            mask_to_save = np.rollaxis(mask_composite, 0, 3)
            scipy.misc.imsave('mask_composite.png', mask_to_save)
        else:
            return
        # change from numpy to pytorch tensor

        mask_composite = torch.from_numpy(mask_composite).float()
        image_composite = torch.from_numpy(image_composite).float()
        image_background_resized = torch.from_numpy(image_background_resized).float()

        feat_tensor = 0

        input_dict = {'label': image_composite, 'inst': mask_composite, 'image': image_background_resized,
                      'feat': feat_tensor, 'path': image_path}
        return input_dict

    def __len__(self):
        return len(IMG_PAIRS)

    def name(self):
        return 'InpaintingDatasetGuided'
