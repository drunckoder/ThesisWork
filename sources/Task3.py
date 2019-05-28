import os
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image as keras_image
import SplitPermutation
from CharacterGenerator import generate_letters

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


def classify_images(images, self_letter):
    result = []
    for index, image in enumerate(images):
        image = np.expand_dims(image / 255., axis=0)
        predict = model.predict(image)[0]
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


def plot_images(images, predictions):
    fig = plt.figure(figsize=(9.6, 5.4))
    letter = None
    for index, img in enumerate(images):
        letter, self_score, top_letter, top_score = predictions[index]
        img = img.astype('int32').transpose((1, 2, 0))
        ax = fig.add_subplot(4, 6, 1 + index, xticks=[], yticks=[])
        title = f'{letter}:{self_score:.1f}'
        if top_score:
            title += f' [{top_letter}:{top_score:.1f}]'
        ax.set_title(title)
        plt.imshow(img)
    fig.tight_layout()
    plt.savefig(fname=f'{OUT_DIR}/{letter}.png')
    plt.show()


def print_table(predictions_list):
    for predictions in predictions_list:
        data = []
        letter = None
        for prediction in predictions:
            letter, score, _, _ = prediction
            data.append(score)
            # print(f'{letter}, {score}')
        _min = np.min(data)
        _max = np.max(data)
        _mean = np.mean(data)
        print(f'{letter} {_min:.2f} {_max:.2f} {_mean:.2f}')


def run():
    images = load_images()
    predictions_list = []
    for index, image in enumerate(images):
        letter = letters[index]
        permutations = SplitPermutation.process(image)
        predictions = classify_images(permutations, letter)
        predictions_list.append(predictions)
        plot_images(permutations, predictions)

    print_table(predictions_list)


run()
