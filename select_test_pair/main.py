"""This is an interactive UI to select test image pairs."""
import cv2
import datetime
import numpy as np
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
cropping = False


def load_img():
    catIds = coco.getCatIds(catNms=['bus', 'person', 'car'])
    imgIds = coco.getImgIds(catIds=catIds)
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    image_info = coco.loadImgs(ids=img['id'])[0]
    image = io.imread(image_info['coco_url'])
    annIds = coco.getAnnIds(imgIds=image_info['id'],
                            areaRng=[2000, float('inf')], iscrowd=None)
    anns = coco.loadAnns(annIds)
    return image, anns, image_info['id']


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed

    if event == cv2.EVENT_LBUTTONDOWN:
        bg_img_copy = bg_img.copy()
        refPt = [(x, y)]
        cropping = True
        cv2.imshow("bg_img", bg_img_copy)

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping is True:
            bg_img_copy = bg_img.copy()
            cv2.rectangle(bg_img_copy, refPt[0], (x, y), (0, 255, 0), 0)
            cv2.imshow("bg_img", bg_img_copy)

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        bg_img_copy = bg_img.copy()
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        cv2.rectangle(bg_img_copy, refPt[0], refPt[1], (0, 255, 0), 0)
        cv2.imshow("bg_img", bg_img_copy)
    return refPt


if __name__ == '__main__':
    c = 0
    np.random.seed(0)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    f = open('result_{}'.format(current_time), 'w')
    while True:
        object_img, object_img_anns, obj_img_id = load_img()
        cv2.imshow('object_img', object_img)
        c = cv2.waitKey(0)
        if c == 13:  # Enter
            object_image_height, object_image_width, _ = object_img.shape
            print('num_of_objects: {}'.format(len(object_img_anns)))
            for i in range(len(object_img_anns)):
                mask = coco.annToMask(object_img_anns[i])
                cv2.imshow('object_img_mask', mask * 255)
                c = cv2.waitKey(0)
                if c == 13:  # Enter
                    object_id = i
                    break
            np.random.seed(0)
            while True:
                bg_img, _, bg_img_id = load_img()
                bg_img = scipy.misc.imresize(bg_img, [256, 256])
                cv2.imshow('bg_img', bg_img)
                c = cv2.waitKey(0)

                if c == 13:  # Enter to select the background image
                    cv2.setMouseCallback("bg_img", click_and_crop)
                    while True:
                        cv2.imshow("bg_img", bg_img)
                        c2 = cv2.waitKey(0)
                        if c2 == 13:
                            mask[mask > 0] = 1  # fineSize x fineSize
                            a = np.where(mask != 0)
                            bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(
                                a[1])

                            object_x = bbox[2]
                            object_y = bbox[0]
                            object_height = bbox[1] - object_y + 1
                            object_width = bbox[3] - object_x + 1

                            new_object_height = refPt[1][1] - refPt[0][1]
                            new_object_width = refPt[1][0] - refPt[0][0]
                            object_image_resize_height = int(
                                object_image_height * new_object_height / object_height)
                            object_image_resize_width = int(
                                object_image_width * new_object_width / object_width)

                            image_resized = scipy.misc.imresize(object_img,
                                                                [
                                                                    object_image_resize_height,
                                                                    object_image_resize_width])

                            mask_resized = scipy.misc.imresize(mask,
                                                               [
                                                                   object_image_resize_height,
                                                                   object_image_resize_width])
                            mask_resized[mask_resized > 0] = 1  # fineSize x fineSize
                            a = np.where(mask_resized != 0)
                            bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(
                                a[1])
                            mask_resized = np.tile(mask_resized, (3, 1, 1))
                            mask_resized = np.rollaxis(mask_resized, 0, 3)

                            object_x = bbox[2]
                            object_y = bbox[0]
                            object_height = new_object_height
                            object_width = new_object_width

                            object_image = image_resized[
                                           object_y:object_y + object_height,
                                           object_x:object_x + object_width, :]
                            object_mask = mask_resized[
                                          object_y:object_y + object_height,
                                          object_x:object_x + object_width, :]
                            object_image_with_background = np.copy(object_image)
                            object_image[object_mask == 0] = 0

                            image_composite = np.copy(bg_img)
                            new_object_y = refPt[0][1]
                            new_object_x = refPt[0][0]
                            image_composite[new_object_y: new_object_y + object_height,
                            new_object_x: new_object_x + object_width,
                            :] = object_image_with_background

                            cv2.imshow("img_composite", image_composite)

                            c3 = cv2.waitKey(0)
                            if c3 == 13:
                                f.write(
                                    '{\n' +
                                    '\'object_img_id\': {},\n'.format(obj_img_id) +
                                    '\'background_img_id\': {}, \n'.format(bg_img_id) +
                                    '\'object_id\': {}, \n'.format(object_id) +
                                    '\'object_composite_x\': {}, \n'.format(
                                        new_object_x) +
                                    '\'object_composite_y\': {}, \n'.format(
                                        new_object_y) +
                                    '\'object_composite_width\': {}, \n'.format(
                                        object_width) +
                                    '\'object_composite_height\': {}, \n}},\n'.format(object_height))
                                break
                    break
        elif c == 113:  # 'q'
            break

    f.close()
