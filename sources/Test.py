import tensorflow
print(tensorflow.pywrap_tensorflow.IsMklEnabled())

import random
from matplotlib import pyplot
import numpy as np
from keras import backend as K
from keras.datasets import cifar10
from CharacterGenerator import generate_data_set


# if K.backend() == 'tensorflow':
#     K.set_image_dim_ordering("th")

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# (m_x_train, m_y_train), (m_x_test, m_y_test), class_names = generate_data_set(train_repeat=5000, test_repeat=1000)
#
# num_classes = 10
#
# # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
# #                'dog', 'frog', 'horse', 'ship', 'truck']
#
# fig = pyplot.figure(figsize=(2, 2))
# ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
# ax.axis('off')
# idx = np.where(y_train[:] == 9)[0]
# features_idx = x_train[idx, ::]
# img_num = 0
# im_tmp = features_idx[img_num, ::]
# im = np.transpose(im_tmp, (1, 2, 0))
# ax.set_title(class_names[9])
# pyplot.imshow(im)
# pyplot.show()
#
# # y_train = np_utils.to_categorical(y_train, num_classes)
# # y_test = np_utils.to_categorical(y_test, num_classes)
# # x_train = x_train.astype('float32')
# # x_test = x_test.astype('float32')
# # x_train /= 255
# # x_test /= 255
