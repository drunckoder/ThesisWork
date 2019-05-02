import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from CharacterGenerator import generate_letters
from NoiseGenerator import add_noise

if K.backend() == 'tensorflow':
    K.set_image_dim_ordering("th")

model = load_model('Task2.h5')

letters = [c for c in generate_letters()]


def load_images():
    result = []
    for letter in letters:
        img = image.load_img('gen/{}.png'.format(letter))
        img = image.img_to_array(img)
        img.reshape((1,) + img.shape)
        img = img.reshape((1,) + img.shape)
        result.append(img)
    return np.array(result)


def classify_image(target_image, letter):
    target_image = target_image / 255.
    predict = model.predict(target_image)
    print('Cls {}'.format(letter), end='  ')
    print('Acc:{:.1f}'.format(predict[0][[letters.index(letter)]][0]), end='  ')
    for i, p in enumerate(predict[0]):
        print('{}:{:.2f}'.format(letters[i], p), end='  ', flush=True)
    print()


def plot_images(_images):
    fig = plt.figure(figsize=(8, 3))
    for i, img in enumerate(_images):
        img = img[0].astype('int32').transpose((1, 2, 0))
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        ax.set_title(letters[i])
        plt.imshow(img)
    plt.show()


np.random.seed(42)

for c in range(10):
    images = load_images()
    noise_level = c / 10.
    print('Noise level: {}'.format(noise_level))
    add_noise(images, noise_level=noise_level)
    plot_images(images)

    for x, im in enumerate(images):
        classify_image(im, letters[x])
