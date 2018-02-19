"""Dataset for guided inpainting test using both the inpainting and
harmonization models. """
import numpy as np
import scipy
import torch

from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from pycocotools.coco import COCO

IMG_PAIRS = [
    {
        'object_img_id': 5037,
        'background_img_id': 447342,
        'object_id': 0,
        'object_composite_x': 5,
        'object_composite_y': 127,
        'object_composite_width': 115,
        'object_composite_height': 122,
    },
    {
        'object_img_id': 429109,
        'background_img_id': 334371,
        'object_id': 9,
        'object_composite_x': 70,
        'object_composite_y': 180,
        'object_composite_width': 107,
        'object_composite_height': 60,
    },
    {
        'object_img_id': 177893,
        'background_img_id': 334371,
        'object_id': 2,
        'object_composite_x': 30,
        'object_composite_y': 150,
        'object_composite_width': 185,
        'object_composite_height': 84,
    }
]


# noinspection PyMethodMayBeStatic
class InpaintingDatasetTest(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.annFile = opt.ann_path
        self.coco = COCO(self.annFile)
        self.dataset_size = len(IMG_PAIRS)

    def load_coco_image(self, image_id, object_id=None):
        image_info = self.coco.loadImgs(ids=image_id)[
            0]
        image_url = image_info['coco_url']
        image_url_split = image_url.split('/')
        image_path = '{}/{}'.format(self.root, image_url_split[-1])
        image = scipy.misc.imread(image_path, mode='RGB')

        mask = None
        if object_id is not None:
            annIds = self.coco.getAnnIds(imgIds=image_info['id'], iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            mask = self.coco.annToMask(anns[object_id])
        return image, mask, image_path

    def compute_bounding_box(self, object_mask):
        object_mask[object_mask > 0] = 1  # fineSize x fineSize
        object_pixels = np.where(object_mask != 0)
        bbox = np.min(object_pixels[0]), np.max(object_pixels[0]), \
               np.min(object_pixels[1]), np.max(object_pixels[1])

        object_x = bbox[2]
        object_y = bbox[0]
        object_height = bbox[1] - object_y + 1
        object_width = bbox[3] - object_x + 1

        return object_x, object_y, object_height, object_width

    def __getitem__(self, index):
        # load object image and mask
        object_image, object_mask, image_path = \
            self.load_coco_image(image_id=IMG_PAIRS[index]['object_img_id'],
                                 object_id=IMG_PAIRS[index]['object_id'])

        background_image, _, _ = \
            self.load_coco_image(image_id=IMG_PAIRS[index]['background_img_id'])

        object_image_height, object_image_width, _ = \
            object_image.shape

        # find bounding box
        _, _, object_ori_height, object_ori_width = \
            self.compute_bounding_box(object_mask)
        print('idnex: {}'.format(index))
        print('height: {}'.format(object_ori_height))
        print('width: {}'.format(object_ori_width))
        # Compute the new size of the image based on the size of inpainted
        # object.
        object_image_resize_height = int(
            object_image_height * IMG_PAIRS[index][
                'object_composite_height'] / object_ori_height)
        object_image_resize_width = int(
            object_image_width * IMG_PAIRS[index][
                'object_composite_width'] / object_ori_width)

        print('idnex: {}'.format(index))
        print('height: {}'.format(object_image_resize_height))
        print('width: {}'.format(object_image_resize_width))

        # Inpainting
        object_image_resized = scipy.misc.imresize(object_image,
                                                   [object_image_resize_height,
                                                    object_image_resize_width])
        object_image_resized_chw = np.rollaxis(object_image_resized, 2,
                                               0)  # 3 x fineSize x fineSize

        # resize object mask
        mask_resized = scipy.misc.imresize(object_mask,
                                           [object_image_resize_height,
                                            object_image_resize_width])
        mask_resized[mask_resized > 0] = 1  # fineSize x fineSize
        # find bounding box
        object_x, object_y, object_height, object_width = \
            self.compute_bounding_box(mask_resized)
        mask_resized = np.tile(mask_resized, (3, 1, 1))

        print('object_height: {}'.format(object_height))
        print('object_width: {}'.format(object_width))

        # normalize object image
        object_image_resized_chw = object_image_resized_chw / 122.5 - 1

        # get the image patch that contains the object.
        object_image_patch_with_bg = object_image_resized_chw[:,
                             object_y:object_y + object_height,
                             object_x:object_x + object_width]  # noqa 501
        object_mask_patch = mask_resized[:, object_y:object_y + object_height,
                                         object_x:object_x + object_width]
        object_image_patch_no_bg = np.copy(object_image_patch_with_bg)
        object_image_patch_no_bg[object_mask_patch == 0] = 0

        # resize and normalize the background image.
        background_image_resized = scipy.misc.imresize(background_image,
                                                       [self.opt.fineSize,
                                                        self.opt.fineSize])  # fineSize x fineSize x 3  # noqa 501
        background_image_resized_chw = np.rollaxis(background_image_resized, 2,
                                                   0)  # noqa 501
        background_image_resized_chw = background_image_resized_chw / 122.5 - 1

        # image composition. We remove the background of the image patch.
        new_object_x = IMG_PAIRS[index]['object_composite_x']
        new_object_y = IMG_PAIRS[index]['object_composite_y']
        image_composite_no_bg = np.copy(background_image_resized_chw)
        image_composite_no_bg[:, new_object_y: new_object_y + object_height,
                        new_object_x: new_object_x + object_width] = object_image_patch_no_bg  # noqa 501

        # Image composition. We keep the background of the image patch.
        image_composite_with_bg = np.copy(background_image_resized_chw)
        image_composite_with_bg[:, new_object_y: new_object_y + object_height, new_object_x: new_object_x + object_width] = object_image_patch_with_bg  # noqa 501

        mask_composite = np.zeros(image_composite_no_bg.shape)
        mask_composite[:, new_object_y:new_object_y + object_height, new_object_x:new_object_x + object_width] = 1 - object_mask_patch  # noqa 501

        mask_composite_object = np.zeros(image_composite_no_bg.shape)
        mask_composite_object[:, new_object_y:new_object_y + object_height,
                              new_object_x:new_object_x + object_width] = \
            object_mask_patch

        mask_composite = torch.from_numpy(mask_composite).float()
        image_composite_no_bg = torch.from_numpy(image_composite_no_bg).float()
        background_image_resized = torch.from_numpy(
            background_image_resized).float()

        feat_tensor = 0

        input_dict = {'input': image_composite_no_bg, 'mask': mask_composite,
                      'image': background_image_resized,
                      'feat': feat_tensor, 'path': image_path,
                      'image_composite_with_bg': image_composite_with_bg,
                      'mask_composite_object': mask_composite_object}

        return input_dict

    def __len__(self):
        return len(IMG_PAIRS)

    def name(self):
        return 'InpaintingDatasetGuided'
