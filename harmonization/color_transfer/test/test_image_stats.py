"""This is to test the color transfer between two COCO images."""
import cv2
import numpy as np
import scipy.misc

image_path = '../images/autumn.jpg'


def image_stats(image, mask=None):
    # compute the mean and standard deviation of each channel
    # When mask == 1, it means the pixel is background and should be ignored.
    # When mask == 0, it means the pixel is the foreground.
    (l, a, b) = cv2.split(image)
    if mask is not None:
        mask = mask.astype(bool)
    l = np.ma.array(l, mask=mask)
    a = np.ma.array(a, mask=mask)
    b = np.ma.array(b, mask=mask)

    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return lMean, lStd, aMean, aStd, bMean, bStd


if __name__ == '__main__':
    image = scipy.misc.imread(image_path, mode='RGB')
    mask = np.ones((image.shape[0], image.shape[1]))
    print(image_stats(image))
    print(image_stats(image, mask))
