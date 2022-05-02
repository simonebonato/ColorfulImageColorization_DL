import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import tensorflow as tf

import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from adam_class import AdamWeightDecayOptimizer
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay


class CNN:
    def __init__(self):
        print('-- Creating CNN model --')
        # self.load_images_from_folder('data')
        self.model = Sequential()
        self.model.add(Input(
            shape=(224, 224, 1),
            # batch_size=None,
            name='input',
        ))
        # layer structure (X, C, S, D, BN, P)
        # see Table 4 - page 24
        # X : output size after applying this operation
        # C : number of filters
        # S : stride
        # D : kernel dilation
        # BN : Batch norm (True or False)
        # Padding : True - same, False - valid
        conv_layers = (
            ('conv1_1', 224, 64, 1, 1, False, True),
            ('conv1_2', 112, 64, 2, 1, True, False),

            ('conv2_1', 112, 128, 1, 1, False, True),
            ('conv2_2', 56, 128, 2, 1, True, False),

            ('conv3_1', 56, 256, 1, 1, False, True),
            ('conv3_2', 56, 256, 1, 1, False, True),
            ('conv3_3', 28, 256, 2, 1, True, False),

            ('conv4_1', 28, 512, 1, 1, False, True),
            ('conv4_2', 28, 512, 1, 1, False, True),
            ('conv4_3', 28, 512, 1, 1, True, True),

            ('conv5_1', 28, 512, 1, 2, False, True),
            ('conv5_2', 28, 512, 1, 2, False, True),
            ('conv5_3', 28, 512, 1, 2, True, True),

            ('conv6_1', 28, 512, 1, 2, False, True),
            ('conv6_2', 28, 512, 1, 2, False, True),
            ('conv6_3', 28, 512, 1, 2, True, True),

            ('conv7_1', 28, 256, 1, 1, False, True),
            ('conv7_2', 28, 256, 1, 1, False, True),
            ('conv7_3', 28, 256, 1, 1, True, True),

            ('conv8_1', 56, 128, .5, 1, False, False),
            ('conv8_2', 56, 128, 1, 1, False, True),
            ('conv8_3', 56, 128, 1, 1, False, True),

        )

        for i, (label, X, C, S, D, BN, P) in enumerate(conv_layers):
            if S >= 1:

                self.model.add(Conv2D(
                    filters=C,
                    kernel_size=(1, 1),
                    strides=S,
                    dilation_rate=D,
                    activation='relu',
                    padding='same' if P else 'valid',
                    name=label
                ))
            else:
                self.model.add(Conv2DTranspose(
                    filters=C,
                    kernel_size=(1, 1),
                    strides=int(1 / S),
                    dilation_rate=D,
                    activation='relu',
                    padding='same' if P else 'valid',
                    name=label
                ))

            if BN:
                self.model.add(BatchNormalization(name=f'BN_{label[4]}'))

        lr = ExponentialDecay(initial_learning_rate=3e-5, decay_steps=10, decay_rate=0.01)
        adam_weight = AdamWeightDecayOptimizer(beta_1=0.9, beta_2=0.99, learning_rate=lr, weight_decay_rate=10**-3)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam_weight)
        print(self.model.summary())

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
                # img = self.scale_resize_image(img)
                images.append(img)
        self.images = images

    def scale_resize_image(self, image):
        image = tf.image.convert_image_dtype(image, tf.float32)  # equivalent to dividing image pixels by 255
        image = tf.image.resize(image, (224, 224))  # Resizing the image to 224x224 dimention
        return image

    def plot_image(self, orig_img):
        img = cv2.cvtColor(orig_img, cv2.COLOR_Lab2BGR)
        cv2.imshow('here is your fuck*ng image', img)
        cv2.waitKey(0)
        # img = cv2.cvtColor(orig_img, cv2.COLOR_Lab2RGB)
        # plt.imshow(img / 255.0)
