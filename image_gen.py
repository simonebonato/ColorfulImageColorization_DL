import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from skimage import io, color


def myFunc(image):
    return color.rgb2lab(np.float32(image) / 255.)


# def plot_image(img):
#     img = color.lab2rgb(img)
#     plt.imshow(img)
#     plt.show()


def custom_data_gen(path):
    datagen = ImageDataGenerator(preprocessing_function=myFunc)
    generator = datagen.flow_from_directory(directory=path,
                                            target_size=(224, 224),
                                            class_mode=None,
                                            batch_size=1,
                                            shuffle=False)
    return generator
