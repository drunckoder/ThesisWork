import numpy as np


def put_pixel(target: np.array, x: int, y: int, color: tuple = (0., 0., 0.)):
    for i in range(3):
        target[i][y][x] = color[i]


def noise_image(target: np.array, noise_level: float, size: int = 32):
    for y in range(size):
        for x in range(size):
            if np.random.rand() < noise_level:
                put_pixel(target=target[0], x=x, y=y)


def add_noise(target: np.array, noise_level: float, size: int = 32):
    for image in target:
        noise_image(target=image, noise_level=noise_level, size=size)
