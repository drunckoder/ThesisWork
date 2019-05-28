import os
import math
import numpy as np
from PIL import Image

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image as keras_image

if K.backend() == 'tensorflow':
    K.set_image_dim_ordering("th")

model = load_model('Task2.h5')

image = keras_image.load_img(f'Test.bmp')
image = keras_image.img_to_array(image)
image = np.expand_dims(image, axis=0)
predict = model.predict(image)[0]

print(predict)
