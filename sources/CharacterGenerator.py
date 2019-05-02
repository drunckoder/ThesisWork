import math
import numpy as np
from string import ascii_uppercase
from PIL import ImageFont, ImageDraw, Image


def render_character(text: str, size: int = 32) -> Image:
    image = Image.new(mode='RGB', size=(size, size), color='white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 32)
    w, h = draw.textsize(text, font=font)
    draw.text((math.ceil((size - w) / 2), math.floor((size - h) / 2) - 2), text, font=font, fill='black')
    image.save(fp='gen/{}.png'.format(text))
    return image


def generate_characters(letters: list, size: int = 32) -> np.array:
    result = []
    for letter in letters:
        image = np.array(render_character(letter, size=size))
        result.append(image.transpose((2, 0, 1)))
    return np.array(result)


def generate_letters(count: int = 10) -> list:
    if not 0 < count <= 26:
        raise ValueError("count must be in range (0, 26]")
    return ascii_uppercase[:count]


def generate_data_set(letters_count: int = 10, size: int = 32, train_repeat: int = 1, test_repeat: int = 1):
    letters = generate_letters(count=letters_count)
    characters = generate_characters(letters, size=size)

    train_size = letters_count * train_repeat
    train_shuffle = np.random.permutation(train_size)
    x_train = characters.repeat(train_repeat, 0)[train_shuffle]
    y_train = np.arange(letters_count).repeat(train_repeat)[train_shuffle].reshape(train_size, 1)

    test_size = letters_count * test_repeat
    test_shuffle = np.random.permutation(test_size)
    x_test = characters.repeat(test_repeat, 0)[test_shuffle]
    y_test = np.arange(letters_count).repeat(test_repeat)[test_shuffle].reshape(test_size, 1)

    return (x_train, y_train), (x_test, y_test), [c for c in letters]
