import os
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image as keras_image
from CharacterGenerator import generate_letters
from NoiseGenerator import add_noise

if K.backend() == 'tensorflow':
    K.set_image_dim_ordering("th")

OUT_DIR = f'{os.path.basename(__file__)[:-3]}_out'
if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)

W_DIR = 'Task2_out'

model = load_model('Task2.h5')

letters = list(generate_letters())


def load_images():
    result = []
    for letter in letters:
        image = keras_image.load_img(f'{W_DIR}/{letter}.bmp')
        image = keras_image.img_to_array(image)
        result.append(image)
    return np.asarray(result)


def classify_image(image, letter):
    image = np.expand_dims(image / 255., axis=0)
    predict = model.predict(image)
    print(f'Cls {letter}', end='  ')
    print(f'Acc:{predict[0][[letters.index(letter)]][0]:.1f}', end='  ')
    for index, p in enumerate(predict[0]):
        print(f'{letters[index]}:{p:.2f}', end='  ', flush=True)
    print()


def plot_images(images, noise_level):
    fig = plt.figure(figsize=(8, 3))
    for index, image in enumerate(images):
        image = image.astype('int32').transpose((1, 2, 0))
        ax = fig.add_subplot(2, 5, 1 + index, xticks=[], yticks=[])
        ax.set_title(letters[index])
        plt.imshow(image)
    plt.savefig(fname=f'{OUT_DIR}/{noise_level}.png')
    plt.show()


def run():
    np.random.seed(42)

    for i in range(10):
        images = load_images()
        noise_level = i / 10.
        print(f'Noise level: {noise_level}')
        add_noise(target=images, noise_level=noise_level)
        plot_images(images, noise_level)

        for x, im in enumerate(images):
            classify_image(im, letters[x])


run()
