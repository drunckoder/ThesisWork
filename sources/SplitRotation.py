from random import shuffle
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


def rotate(image: np.array, k: int):
    return np.rot90(image, k=k, axes=(1, 2))


def mutate_image(image: np.array, rotation: tuple, size: int = 32):
    def concat(im1, im2, axis):
        _result = []
        for i in range(len(im1)):
            _result.append(np.concatenate((im1[i], im2[i]), axis))
        return np.asarray(_result)

    ul = rotate(image[up_left(size)], rotation[0])
    ur = rotate(image[up_right(size)], rotation[1])
    dl = rotate(image[down_left(size)], rotation[2])
    dr = rotate(image[down_right(size)], rotation[3])

    up = concat(ul, ur, axis=1)
    dn = concat(dl, dr, axis=1)
    result = concat(up, dn, axis=0)

    return result


def generate_angles():
    return [(w, x, y, z) for w in range(4) for x in range(4) for y in range(4) for z in range(4)]


def process(image: np.array, size: int = 32):
    result = []
    angles_list = generate_angles()
    # shuffle(angles_list)
    # angles_list = angles_list[:32]
    for angles in angles_list:
        angles = tuple(map(int, angles))
        result.append(mutate_image(image=image, rotation=angles, size=size))
    result = np.asarray(result)
    return result
