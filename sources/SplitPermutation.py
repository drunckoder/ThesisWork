from itertools import permutations

import numpy as np


def up_left(size: int = 32):
    half = size // 2
    return slice(None), slice(0, half), slice(0, half)


def up_right(size: int = 32):
    half = size // 2
    return slice(None), slice(0, half), slice(half, size)


def down_left(size: int = 32):
    half = size // 2
    return slice(None), slice(half, size), slice(0, half)


def down_right(size: int = 32):
    half = size // 2
    return slice(None), slice(half, size), slice(half, size)


def mutate_image(image: np.array, order: tuple, size: int = 32):
    def concat(im1, im2, axis):
        _result = []
        for i in range(len(im1)):
            _result.append(np.concatenate((im1[i], im2[i]), axis))
        return np.array(_result)

    parts = [
        (image[up_left(size)], order[0]),
        (image[up_right(size)], order[1]),
        (image[down_left(size)], order[2]),
        (image[down_right(size)], order[3])
    ]

    parts = sorted(parts, key=lambda x: x[1])

    up = concat(parts[0][0], parts[1][0], axis=1)
    dn = concat(parts[2][0], parts[3][0], axis=1)
    result = concat(up, dn, axis=0)

    return result


def process(targets: np.array, size: int = 32):
    result = []
    for image in targets:
        result.append([process_image(target=image[0], size=size)])
    result = np.array(result)
    return result
