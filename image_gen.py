import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

from skimage import io, color


def get_L(image):
    lab = color.rgb2lab(np.float32(image) / 255.)
    return lab


def get_ab(image):
    lab = color.rgb2lab(np.float32(image) / 255.)
    return lab


def plot_image_from_Lab(img, grayscale=False):
    """if we want to plot only the L channel
    then we have to set a and b to zero
    then convert to rgb and plot!"""
    if grayscale:
        ab = img[:, :, 1:]
        img[:, :, 1:] = 0 * ab
    img = color.lab2rgb(img)
    plt.imshow(img)
    plt.show()


def custom_data_gen(path, img_size=256, batch_size=1):
    x_datagen = ImageDataGenerator(preprocessing_function=get_L)
    x_generator = x_datagen.flow_from_directory(directory=path,
                                                target_size=(img_size, img_size),
                                                batch_size=batch_size,
                                                shuffle=False,
                                                classes=None)

    y_datagen = ImageDataGenerator(preprocessing_function=get_ab)
    y_generator = y_datagen.flow_from_directory(directory=path,
                                                target_size=(img_size, img_size),
                                                batch_size=batch_size,
                                                shuffle=False,
                                                classes=None)
    generator = zip(x_generator, y_generator)
    return generator


# def load_images_from_folder(self, folder):
#     print('-- Reading images --')
#     images = []
#     for filename in os.listdir(folder)[:500]:
#         img = cv2.imread(os.path.join(folder, filename))
#         if img is not None:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
#             # img = self.scale_resize_image(img)
#             images.append(img)
#     self.images = images


g = custom_data_gen('data/train')
for image, label in g:
    plot_image_from_Lab(image[0][0])
    # print(label)
    break
