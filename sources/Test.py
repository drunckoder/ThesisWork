import numpy
import math
import cv2

import os

os.chdir('Task6_out')

for x in range(0, 9):
    original = cv2.imread(f'0.0.bmp')
    contrast = cv2.imread(f'0.{x}.bmp', 1)

    PIXEL_MAX = 255.0


    def psnr(img1, img2):
        mse = numpy.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


    d = psnr(original, contrast)
    print(d)

