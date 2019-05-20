import os
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image as keras_image
from CharacterGenerator import generate_letters
from NoiseGenerator import add_noise
from copy import deepcopy

if K.backend() == 'tensorflow':
    K.set_image_dim_ordering("th")

OUT_DIR = f'{os.path.basename(__file__)[:-3]}_out'
if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)

W_DIR = 'Task2_out'

POSITIVE_THRESHOLD = 0.5

model = load_model('Task2.h5')
letters = list(generate_letters())


def load_images():
    result = []
    for letter in letters:
        image = keras_image.load_img(f'{W_DIR}/{letter}.bmp')
        image = keras_image.img_to_array(image)
        result.append(image)
    return np.asarray(result)


def classify_images(images):
    result = []
    for index, image in enumerate(images):
        image = np.expand_dims(image / 255., axis=0)
        predict = model.predict(image)[0]
        self_letter = letters[index]
        self_index = letters.index(self_letter)
        self_score = predict[self_index]

        print(f'Cls {self_letter}', end='  ')
        print(f'Acc:{self_score:.1f}', end='  ')
        for ix, prediction in enumerate(predict):
            print(f'{letters[ix]}:{prediction:.1f}', end='  ', flush=True)
        print()

        top_index = int(np.argmax(predict))
        top_letter = None
        top_score = None
        if self_index != top_index and predict[top_index] > POSITIVE_THRESHOLD:
            top_letter = letters[top_index]
            top_score = predict[top_index]

        result.append((self_letter, self_score, top_letter, top_score))
    return result


def plot_images(images, predictions, noise_level):
    fig = plt.figure(figsize=(8, 3))
    for index, img in enumerate(images):
        letter, self_score, top_letter, top_score = predictions[index]
        img = img.astype('int32').transpose((1, 2, 0))
        ax = fig.add_subplot(2, 5, 1 + index, xticks=[], yticks=[])
        title = f'{letter}:{self_score:.1f}'
        if top_score:
            title += f' [{top_letter}:{top_score:.1f}]'
        ax.set_title(title)
        plt.imshow(img)
    fig.tight_layout()
    plt.savefig(fname=f'{OUT_DIR}/{noise_level}.png')
    plt.show()


def run():
    np.random.seed(1337)
    original_images = load_images()
    for i in range(10):
        images = deepcopy(original_images)
        noise_level = i / 10.
        print(f'Noise level: {noise_level}')
        add_noise(target=images, noise_level=noise_level)
        predictions = classify_images(images)
        plot_images(images, predictions, noise_level)


run()
