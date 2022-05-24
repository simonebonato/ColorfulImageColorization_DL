import cv2
import tensorflow.keras.utils
import numpy as np
from PIL import Image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
from loss_function import *
from skimage import io, color


def get_L(image):
    lab = color.rgb2lab(np.float32(image) / 255.)
    return lab


def get_ab(image):
    lab = color.rgb2lab(np.float32(image) / 255.)
    return lab


def plot_image_from_Lab(img, grayscale=False, from_L=False, gt=False, savename=None):
    """if we want to plot only the L channel
    then we have to set a and b to zero
    then convert to rgb and plot!"""
    if from_L:
        temp = np.zeros((img.shape[0], img.shape[1], 3))
        temp[:, :, 0] = img[:, :, 0]
        img = temp
    if grayscale:
        ab = img[:, :, 1:]
        img[:, :, 1:] = 0 * ab
    img = color.lab2rgb(img)
    plt.imshow(img)
    if not gt:
        plt.savefig(savename)
        plt.show()


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder)[:500]:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = img.astype("float32") / 255
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            img = cv2.resize(img, (256, 256))
            images.append(img)
    return images

def get_partitions(train_path, val_path):
    from os import listdir
    from os.path import isfile, join
    train_imgs = [join(train_path, f) for f in listdir(train_path) if isfile(join(train_path, f))]
    val_imgs = [join(val_path, f) for f in listdir(val_path) if isfile(join(val_path, f))]
    return train_imgs, val_imgs


class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, partition, batch_size=1, dim=(256, 256), n_channels=(1, 2), shuffle=True):
        '''
        partition: list with paths of images
        dim: desired image dimension
        channels: number of channels in X and Y
        '''
        'Initialization'
        self.dim = dim
        self.out_dim = (dim[0] // 4, dim[1] // 4)
        self.batch_size = batch_size
        self.partition = partition
        self.n_channels = n_channels
        self.shuffle = shuffle
        # Load the array of quantized ab value

        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.partition))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, imgs_path_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels[0]))
        Y = np.empty((self.batch_size, *self.out_dim, self.n_channels[1]))

        # Generate data
        for i, ID in enumerate(imgs_path_temp):
            img = cv2.imread(ID)
            img = img.astype("float32") / 255
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            img = cv2.resize(img, self.dim)
            X[i, :, :, 0] = img[:, :, 0]
            out_img = cv2.resize(img, self.out_dim, cv2.INTER_CUBIC)
            Y[i, :, :] = out_img[:, :, 1:]
        return X, Y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.partition) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        partition_temp = [self.partition[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(partition_temp)

        return X, Y


# train_path = 'data/train'
# val_path = 'data/val'
# partition = {'train': (get_partitions(train_path, val_path))[0], 'val': (get_partitions(train_path, val_path))[1]}
#
# params = {'dim': (256, 256),
#           'batch_size': 10,
#           'n_channels': (1, 2),
#           'shuffle': False}
#
# training_generator = DataGenerator(partition['train'], **params)
#
# for i in training_generator:
#     print(i[0].shape)
#     print(i[1].shape)
#     break
