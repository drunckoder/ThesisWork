import random

import numpy as np


def up_left(size: int = 32):
    half = size // 2
    return slice(0, half), slice(0, half)


def up_right(size: int = 32):
    half = size // 2
    return slice(0, half), slice(half, size)


def down_left(size: int = 32):
    half = size // 2
    return slice(half, size), slice(0, half)


def down_right(size: int = 32):
    half = size // 2
    return slice(half, size), slice(half, size)


def apply_transform(target: np.array, k: int):
    return np.rot90(target, k=k)


def process_image(target: np.array, size: int = 32):
    def process_quadrant(clip):
        _result = []
        k = random.randint(1, 3)
        for channel in target:
            _result.append(apply_transform(channel[clip(size)], k))
        return np.array(_result)

    def concat(im1, im2, axis):
        _result = []
        for i in range(len(im1)):
            _result.append(np.concatenate((im1[i], im2[i]), axis))
        return np.array(_result)

    ul = process_quadrant(up_left)
    ur = process_quadrant(up_right)
    dl = process_quadrant(down_left)
    dr = process_quadrant(down_right)

    up = concat(ul, ur, axis=1)
    dn = concat(dl, dr, axis=1)
    result = concat(up, dn, axis=0)

    return result


def process(targets: np.array, size: int = 32):
    result = []
    for image in targets:
        result.append([process_image(target=image[0], size=size)])
    result = np.array(result)
    return result
