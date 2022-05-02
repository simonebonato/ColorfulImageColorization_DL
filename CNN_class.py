from keras.models import Sequential
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import numpy as np
from keras.layers import Conv2D, BatchNormalization, Conv2DTranspose
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from adam_class import *



class CNN:
    def __init__(self):
        print('-- Creating CNN model --')
        self.load_images_from_folder('data')
        self.model = Sequential()
        # layer structure (X, C, S, D, Sa, De, BN, L)
        # see Table 4 - page 24
        conv_layers = (
            (224, 64, 1, 1, 1, 1, False, False),
            (112, 64, 2, 1, 1, 1, True, False),

            (112, 128, 1, 1, 2, 2, False, False),
            (56, 128, 2, 1, 2, 2, True, False),

            (56, 256, 1, 1, 4, 4, False, False),
            (56, 256, 1, 1, 4, 4, False, False),
            (28, 256, 2, 1, 4, 4, True, False),

            (28, 512, 1, 1, 8, 8, False, False),
            (28, 512, 1, 1, 8, 8, False, False),
            (28, 512, 1, 1, 8, 8, True, False),

            (28, 512, 1, 2, 8, 16, False, False),
            (28, 512, 1, 2, 8, 16, False, False),
            (28, 512, 1, 2, 8, 16, True, False),

            (28, 512, 1, 2, 8, 16, False, False),
            (28, 512, 1, 2, 8, 16, False, False),
            (28, 512, 1, 2, 8, 16, True, False),

            (28, 256, 1, 1, 8, 8, False, False),
            (28, 256, 1, 1, 8, 8, False, False),
            (28, 256, 1, 1, 8, 8, True, False),

            (56, 128, .5, 2, 4, 4, False, False),
            (56, 128, 1, 2, 4, 4, False, False),
            (56, 128, 1, 1, 4, 4, False, True),

        )

        for (X, C, S, D, Sa, De, BN, L) in conv_layers:
            if S >= 1:
                self.model.add(Conv2D(
                    filters=C,
                    kernel_size=X,
                    strides=S,
                    dilation_rate=D,
                    activation='relu'
                ))
            else:
                self.model.add(Conv2DTranspose(
                    filters=C,
                    kernel_size=X,
                    strides=int(1 / S),
                    dilation_rate=D,
                    activation='relu'
                ))

            if BN:
                self.model.add(BatchNormalization())

        sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)


    def train_val_split(self, train_p):
        pass

    def loss_function(self):
        pass

    def load_images_from_folder(self, folder):
        print('-- Reading images --')
        images = []
        for filename in tqdm(os.listdir(folder)[:500]):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                images.append(img)
        self.images = images

    def plot_image(self, orig_img):
        img = cv2.cvtColor(orig_img, cv2.COLOR_Lab2BGR)
        cv2.imshow('here is your fuck*ng image', img)
        cv2.waitKey(0)
        # img = cv2.cvtColor(orig_img, cv2.COLOR_Lab2RGB)
        # plt.imshow(img / 255.0)
