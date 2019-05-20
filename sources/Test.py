import os
import math
import numpy as np
from PIL import Image

W_DIR = 'Task6_out'

for x in range(0, 9):
    A = np.array(Image.open(f'{W_DIR}/0.0.bmp'))
    B = np.array(Image.open(f'{W_DIR}/0.{x}.bmp'))

    print(A.dtype)

    # print(A[0][0])
    # print(B[0][0])


    def psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        print(f'MSE: {mse}')
        if mse == 0:
            return 100  # undefined
        return 20 * math.log10(255. / math.sqrt(mse))


    d = psnr(A, B)
    print(d)

#
#

# import math
# import numpy as np
#
# import cv2
#
# W_DIR = 'Task6_out'
#
# for x in range(0, 9):
#     A = cv2.imread(f'{W_DIR}/0.0.bmp')
#     B = cv2.imread(f'{W_DIR}/0.{x}.bmp', 1)
#
#     print(A[0][0])
#     print(B[0][0])
#
#
#     def psnr(img1, img2):
#         mse = np.mean((img1 - img2) ** 2)
#         print(f'MSE: {mse}')
#         if mse == 0:
#             return 100  # undefined
#         return 20 * math.log10(255. / math.sqrt(mse))
#
#
#     d = psnr(A, B)
#     print(d)