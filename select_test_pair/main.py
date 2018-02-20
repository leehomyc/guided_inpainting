"""This is an interactive UI to select test image pairs."""
import cv2
import numpy as np
import scipy
import scipy.misc
import skimage.io as io

from pycocotools.coco import COCO

dataDir = '/Users/harry/Documents/data/COCO'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)

catIds = coco.getCatIds(catNms=['bus'])
imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
I = io.imread(img['coco_url'])

refPt = []


def load_object_img():
    catIds = coco.getCatIds(catNms=['bus'])
    imgIds = coco.getImgIds(catIds=catIds)
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    image_info = coco.loadImgs(ids=img['id'])[0]
    image = io.imread(image_info['coco_url'])
    annIds = coco.getAnnIds(imgIds=image_info['id'],
                            areaRng=[2000, float('inf')], iscrowd=None)
    anns = coco.loadAnns(annIds)
    return image, anns


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(bg_img, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("bg_img", bg_img)
    return refPt


if __name__ == '__main__':
    c = 0
    np.random.seed(0)
    while 'q' != chr(c & 255):
        object_img, object_img_anns = load_object_img()
        cv2.imshow('object_img', object_img)
        c = cv2.waitKey(0)

    print(len(object_img_anns))
    for i in range(len(object_img_anns)):
        mask = coco.annToMask(object_img_anns[i])
        print(mask.max())
        cv2.imshow('object_img_mask', mask * 255)
        c = cv2.waitKey(0)
        if 'q' == chr(c & 255):
            break

    np.random.seed(0)
    c = 0
    object_image_height, object_image_width, _ = object_img.shape
    while 'q' != chr(c & 255):
        bg_img, _ = load_object_img()
        bg_img = scipy.misc.imresize(bg_img, [256, 256])
        cv2.imshow('bg_img', bg_img)


        c=cv2.waitKey(0)

        if 'q' == chr(c & 255):

            cv2.setMouseCallback("bg_img", click_and_crop)

            while True:
                cv2.imshow("bg_img", bg_img)
                c = cv2.waitKey(0)
                mask[mask > 0] = 1  # fineSize x fineSize
                a = np.where(mask != 0)
                bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])

                object_x = bbox[2]
                object_y = bbox[0]
                object_height = bbox[1] - object_y + 1
                object_width = bbox[3] - object_x + 1

                print(refPt[1])
                new_object_height = refPt[1][1] - refPt[0][1]
                new_object_width = refPt[1][0] - refPt[0][0]
                object_image_resize_height = int(
                    object_image_height * new_object_height / object_height)
                object_image_resize_width = int(
                    object_image_width * new_object_width / object_width)

                image_resized = scipy.misc.imresize(object_img,
                                                    [object_image_resize_height,
                                                     object_image_resize_width])

                mask_resized = scipy.misc.imresize(mask, [object_image_resize_height,
                                                          object_image_resize_width])
                mask_resized[mask_resized > 0] = 1  # fineSize x fineSize
                a = np.where(mask_resized != 0)
                bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
                mask_resized = np.tile(mask_resized, (3, 1, 1))
                mask_resized = np.rollaxis(mask_resized, 0, 3)

                object_x = bbox[2]
                object_y = bbox[0]
                object_height = new_object_height
                object_width = new_object_width

                object_image = image_resized[object_y:object_y + object_height,
                               object_x:object_x + object_width, :]
                object_mask = mask_resized[object_y:object_y + object_height,
                              object_x:object_x + object_width, :]
                object_image_with_background = np.copy(object_image)
                object_image[object_mask == 0] = 0

                image_composite = np.copy(bg_img)
                new_object_y = refPt[0][1]
                new_object_x = refPt[0][0]
                image_composite[new_object_y: new_object_y + object_height,
                new_object_x: new_object_x + object_width, :] = object_image_with_background

                cv2.imshow("img_composite", image_composite)

                c = cv2.waitKey(0)
                if 'q' == chr(c & 255):
                    break
    # place it
