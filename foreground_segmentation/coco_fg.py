"""This is the file to train mask RCNN for COCO object foreground extraction."""

import numpy as np
import os
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

import skimage.color
import skimage.io

from third_party.Mask_RCNN.config import Config
import third_party.Mask_RCNN.model as modellib
import third_party.Mask_RCNN.utils as utils

ROOT_DIR = os.getcwd()
DEFAULT_DATASET_YEAR = "2017"
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # We only consider foreground objects.
    IMAGE_MIN_DIM = 200
    IMAGE_MAX_DIM = 256


class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=2017,
                  class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO(
            "{}/annotations/instances_{}{}.json".format(dataset_dir, subset,
                                                        year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        # for i in class_ids:
        #    self.add_class("coco", i, coco.loadCats(i)[0]["name"])
        self.add_class("coco", 1, "object")

        # Add images
        img_cnt = 1
        for i in image_ids:
            anns = coco.loadAnns(coco.getAnnIds(
                imgIds=[i], catIds=class_ids,
                areaRng=[2000, float('inf')],
                iscrowd=None))
            for j, ann in enumerate(anns):
                self.add_image(
                    "coco", image_id=img_cnt,
                    path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                    width=coco.imgs[i]["width"],
                    height=coco.imgs[i]["height"],
                    annottion_id=j,
                    ann=ann)
                img_cnt = img_cnt + 1
        if return_coco:
            return coco

    def load_image(self, image_id):
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        image_info = self.image_info[image_id]
        object_mask = self.annToMask(self.image_info[image_id]['ann'],
                                     image_info["height"],
                                     image_info["width"])
        object_mask[object_mask > 0] = 1  # fineSize x fineSize
        object_pixels = np.where(object_mask != 0)
        bbox = np.min(object_pixels[0]), np.max(object_pixels[0]), \
               np.min(object_pixels[1]), np.max(object_pixels[1])

        object_x = bbox[2]
        object_y = bbox[0]
        object_height = bbox[1] - object_y + 1
        object_width = bbox[3] - object_x + 1
        return image[object_y:object_y + object_height,
               object_x:object_x + object_width, :]


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)
        object_mask = self.annToMask(self.image_info[image_id]["ann"],
                                     image_info["height"],
                                     image_info["width"])
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        object_pixels = np.where(object_mask != 0)
        bbox = np.min(object_pixels[0]), np.max(object_pixels[0]), \
               np.min(object_pixels[1]), np.max(object_pixels[1])

        object_x = bbox[2]
        object_y = bbox[0]
        object_height = bbox[1] - object_y + 1
        object_width = bbox[3] - object_x + 1

        object_mask = object_mask[object_y:object_y + object_height,
                      object_x: object_x + object_width]
        instance_masks = [object_mask]
        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array([1], np.int32)
        return mask, class_ids

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=False,
                        default="",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    if args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    else:
        model_path = None

    # Load weights
    if model_path is not None:
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train", year=args.year,
                                auto_download=args.download)
        dataset_train.load_coco(args.dataset, "val", year=args.year,
                                auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        dataset_val.load_coco(args.dataset, "val", year=args.year,
                              auto_download=args.download)
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
