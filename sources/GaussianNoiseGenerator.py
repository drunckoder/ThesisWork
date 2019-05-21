import numpy as np


def add_noise(target: np.array, mean: float, stddev: float):
    for index, image in enumerate(target):
        image += np.random.normal(mean, stddev, image.shape)
        target[index] = np.clip(image, 0, 255)
