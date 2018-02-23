"""Dataset for guided inpainting test using both the inpainting and
harmonization models. """
import numpy as np
import scipy
import torch

from inpainting_harmonization_test.data.img_pairs import IMG_PAIRS
from pycocotools.coco import COCO


def get_segmentation(image, mask, model):
    """

    :param image: The image.
    :param mask: The ground truth mask.
    :param model: The segmentation model
    :return:
    """
    results = model.detect([image], verbose=1)
    r = results[0]
    masks = r['masks']  # height x width x num_objects. The range is [0,1]
    height, width, num_objects = masks.shape
    max_overlap_cnt = 0
    max_overlap_id = -1
    for i in range(num_objects):
        cur_mask = masks[:, :, i]
        overlap = np.logical_and(cur_mask, mask)
        overlap_cnt = np.sum(overlap)
        if overlap_cnt > max_overlap_cnt:
            max_overlap_cnt = overlap_cnt
            max_overlap_id = i

    if max_overlap_id == -1:
        return np.zeros(height, width)
    return masks[:, :, max_overlap_id]


# noinspection PyMethodMayBeStatic
class InpaintingDatasetTest:
    def initialize(self, opt, model):
        self.opt = opt
        self.root = opt.dataroot
        self.annFile = opt.ann_path
        self.coco = COCO(self.annFile)
        self.dataset_size = len(IMG_PAIRS)
        self.model = model

    def load_coco_image(self, image_id, object_id=None, use_seg=False):
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

        if use_seg is True:
            mask = get_segmentation(image, mask, self.model)
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

    def get_item(self, index):
        # load object image and mask

        object_image, object_mask, image_path = \
            self.load_coco_image(image_id=IMG_PAIRS[index]['object_img_id'],
                                 object_id=IMG_PAIRS[index]['object_id'],
                                 use_seg=self.opt.use_segmentation)

        background_image, _, _ = \
            self.load_coco_image(image_id=IMG_PAIRS[index]['background_img_id'])

        object_image_height, object_image_width, _ = \
            object_image.shape

        # find bounding box
        _, _, object_ori_height, object_ori_width = \
            self.compute_bounding_box(object_mask)

        # Compute the new size of the image based on the size of inpainted
        # object.
        object_image_resize_height = int(
            object_image_height * IMG_PAIRS[index][
                'object_composite_height'] / object_ori_height)
        object_image_resize_width = int(
            object_image_width * IMG_PAIRS[index][
                'object_composite_width'] / object_ori_width)

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
        mask_resized_hw = mask_resized.copy()
        object_x, object_y, object_height, object_width = \
            self.compute_bounding_box(mask_resized)
        mask_resized = np.tile(mask_resized, (3, 1, 1))

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
        image_composite_with_bg[:, new_object_y: new_object_y + object_height,
        new_object_x: new_object_x + object_width] = object_image_patch_with_bg  # noqa 501

        mask_composite = np.zeros(image_composite_no_bg.shape)
        mask_composite[:, new_object_y:new_object_y + object_height,
        new_object_x:new_object_x + object_width] = 1 - object_mask_patch  # noqa 501

        mask_composite_object = np.zeros(image_composite_no_bg.shape)
        mask_composite_object[:, new_object_y:new_object_y + object_height,
        new_object_x:new_object_x + object_width] = \
            object_mask_patch

        mask_composite = torch.from_numpy(mask_composite).float()
        image_composite_no_bg = torch.from_numpy(image_composite_no_bg).float()
        background_image_resized_chw = torch.from_numpy(
            background_image_resized_chw).float()
        mask_composite_object = torch.from_numpy(mask_composite_object).float()
        image_composite_with_bg = torch.from_numpy(
            image_composite_with_bg).float()

        feat_tensor = 0

        image_composite_no_bg = image_composite_no_bg.unsqueeze(0)
        mask_composite = mask_composite.unsqueeze(0)
        background_image_resized_chw = background_image_resized_chw.unsqueeze(0)
        image_composite_with_bg = image_composite_with_bg.unsqueeze(0)
        mask_composite_object = mask_composite_object.unsqueeze(0)

        input_dict = {'input': image_composite_no_bg, 'mask': mask_composite,
                      'image': background_image_resized_chw,
                      'feat': feat_tensor, 'path': [image_path],
                      'image_composite_with_bg': image_composite_with_bg,
                      'mask_composite_object': mask_composite_object}

        return input_dict

    def __len__(self):
        return len(IMG_PAIRS)

    def name(self):
        return 'InpaintingDatasetGuided'
