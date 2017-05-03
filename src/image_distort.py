"""Distort operations."""
import cv2
import numpy as np
import random


def print_img(filename, name='test'):
    """Print image for testing."""
    cv2.imshow(name, filename)
    cv2.waitKey(0)


def flip(filename, horizon=True, vertical=True):
    """Filp the image, horizonly by default."""
    src = cv2.imread(filename)
    h = -1 if horizon else 1
    v = -1 if vertical else 1
    res = src[::v, ::h]
    # print_img(res, 'res')
    return res


def rotate(filename, rotate_angle, ratio):
    """Rotate image.

    Args:
        ratio: bigger/smaller ratio
    """
    src = cv2.imread(filename)
    rows, cols = src.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), rotate_angle, ratio)
    res = cv2.warpAffine(src, M, (rows, cols))

    # print_img(res, 'res')
    return res


def translation(filename, dis_x, dis_y):
    """Shift Image.

    Args:
        x: dis_x
        y: dis_y
    """
    src = cv2.imread(filename)
    H = np.float32([[1, 0, dis_x], [0, 1, dis_y]])
    rows, cols = src.shape[:2]
    res = cv2.warpAffine(src, H, (rows, cols))

    # print_img(res, 'res')
    return res


def changeRGB(filename, n):
    """Randomly change the RGB value of n pixels."""
    src = cv2.imread(filename)
    rows, cols = src.shape[:2]
    for k in range(n):  # Create 5000 noisy pixels
        i = random.randint(0, rows-1)
        j = random.randint(0, cols-1)
        color = (random.randrange(256),
                 random.randrange(256),
                 random.randrange(256))
        src[i, j] = color

    # print_img(res, 'res')
    return src


def crop(filename, des_size):
    """Random Crop.

    Args:
        des_size: the size of the target image, for example 224.
    """
    src = cv2.imread(filename)
    rows, cols = src.shape[:2]

    min_side = rows if rows < cols else cols
    if min_side < des_size:
        # if the size of original image is smaller than 224*224
        # actually this is not allowed!
        des = cv2.resize(src, (des_size, des_size))
    else:
        if rows < cols:
            rows0 = des_size
            cols0 = cols * des_size / rows
        else:
            cols0 = des_size
            rows0 = rows * des_size / cols

        des0 = cv2.resize(src, (cols0, rows0))
        i = random.randint(0, rows0 - des_size)
        j = random.randint(0, cols0 - des_size)

        des = des0[i:i+224, j:j+224]  # cut a 224*224 part of the resized image

    # print_img(des, 'res')
    return des


def main():
    """Test Module."""
    flip("../tmp/test.jpg")
    # rotate("../tmp/test.jpg", 180, 1)
    # translation("../tmp/test.jpg", 50, 10)
    # changeRGB("../tmp/test.jpg", 2000)
    # crop("../tmp/test.jpg", 224)


if __name__ == '__main__':
    main()
