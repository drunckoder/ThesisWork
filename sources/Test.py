import os
import math
import numpy as np
from PIL import Image

W_DIR = 'Task6_out'

for x in range(0, 9):
    A = np.array(Image.open(f'{W_DIR}/0.0.bmp'), dtype=np.float64)
    B = np.array(Image.open(f'{W_DIR}/0.{x}.bmp'), dtype=np.float64)


    def psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100  # undefined
        return 20 * math.log10(255. / math.sqrt(mse))


    d = psnr(A, B)
    print(d)

