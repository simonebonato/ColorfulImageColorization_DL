import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from skimage import io, color


def myFunc(image):
    lab = color.rgb2lab(np.float32(image) / 255.)
    # lab = color.rgb2lab(image)
    return lab


def plot_image(img):
    img = color.lab2rgb(img)
    plt.imshow(img)
    # plt.set_cmap('gray')
    plt.show()


if __name__ == '__main__':
    # datagen = ImageDataGenerator(preprocessing_function=myFunc)
    # path = r'C:\Users\simon\PycharmProjects\DeepLearningPython\GroupProject\test'
    # generator = datagen.flow_from_directory(directory=path, target_size=(224, 224), class_mode=None, batch_size=1)

    datagen = ImageDataGenerator(preprocessing_function=myFunc)
    path = r'C:\Users\simon\PycharmProjects\DeepLearningPython\GroupProject\test'
    generator = datagen.flow_from_directory(directory=path,
                                            target_size=(224, 224),
                                            class_mode=None,
                                            batch_size=1,
                                            shuffle=False)

    for i in generator:
        plot_image(i[0])
        break
