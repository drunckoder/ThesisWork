import os
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image as keras_image
import SplitPermutation
from CharacterGenerator import generate_letters

OUT_DIR = f'{os.path.basename(__file__)[:-3]}_out'
if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)

if K.backend() == 'tensorflow':
    K.set_image_dim_ordering("th")

model = load_model('Task2.h5')
letters = list(generate_letters())


def load_images():
    result = []
    for letter in letters:
        img = keras_image.load_img('gen/{}.png'.format(letter))
        img = keras_image.img_to_array(img)
        img.reshape((1,) + img.shape)
        img = img.reshape((1,) + img.shape)
        result.append(img)
    return np.array(result)


def classify_images(images, letter):
    result = []
    for index, image in enumerate(images):
        image = image / 255.
        predict = model.predict(image)
        print('Cls {}'.format(letter), end='  ')
        print('Acc:{:.1f}'.format(predict[0][[letters.index(letter)]][0]), end='  ')
        for ix, prediction in enumerate(predict[0]):
            print('{}:{:.1f}'.format(letters[ix], prediction), end='  ', flush=True)
        print()
        result.append(predict[0][[letters.index(letter)]][0])
    return result


def plot_images(images, titles, letter):
    fig = plt.figure(figsize=(9.6, 5.4))
    for index, img in enumerate(images):
        img = img[0].astype('int32').transpose((1, 2, 0))
        ax = fig.add_subplot(4, 6, 1 + index, xticks=[], yticks=[])
        ax.set_title('{:.1f}'.format(titles[index]))
        plt.imshow(img)
    fig.tight_layout()
    plt.savefig(fname=f'{OUT_DIR}/{letter}.png')
    plt.show()


def run():
    images = load_images()
    for index, image in enumerate(images):
        letter = letters[index]
        permutations = SplitPermutation.process(image)
        predictions = classify_images(permutations, letter)
        plot_images(permutations, predictions, letter)


run()
