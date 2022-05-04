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
            shape=(256, 256, 1),
            batch_size = 1,
            name='input',
        ))

        # see Table 4 - page 24
        # C : number of filters
        # S : stride
        # D : kernel dilation
        # BN : Batch norm (True or False)
        # Padding : True - same, False - valid
        # layer structure ('name', C, S, D, K, BN)
        conv_layers = (
            ('conv1_1', 64, 1, 1, 3, False),
            ('conv1_2', 64, 2, 1, 3, True),

            ('conv2_1', 128, 1, 1, 3, False),
            ('conv2_2', 128, 2, 1, 3, True),

            ('conv3_1', 256, 1, 1, 3, False),
            ('conv3_2', 256, 1, 1, 3, False),
            ('conv3_3', 256, 2, 1, 3, True),

            ('conv4_1', 512, 1, 1, 3, False),
            ('conv4_2', 512, 1, 1, 3, False),
            ('conv4_3', 512, 1, 1, 3, True),

            ('conv5_1', 512, 1, 2, 3, False),
            ('conv5_2', 512, 1, 2, 3, False),
            ('conv5_3', 512, 1, 2, 3, True),

            ('conv6_1', 512, 1, 2, 3, False),
            ('conv6_2', 512, 1, 2, 3, False),
            ('conv6_3', 512, 1, 2, 3, True),

            ('conv7_1', 512, 1, 1, 3, False),
            ('conv7_2', 512, 1, 1, 3, False),
            ('conv7_3', 512, 1, 1, 3, True),

            ('conv8_1', 128, .5, 1, 4, False),
            ('conv8_2', 128, 1, 1, 3, False),
            ('conv8_3', 128, 1, 1, 3, False),

            ('conv_out', 313, 1, 1, 1, False)

        )

        for i, (label, C, S, D, K, BN) in enumerate(conv_layers):
            if S >= 1:
                self.model.add(Conv2D(
                    filters=C,
                    kernel_size=(K, K),
                    strides=S,
                    dilation_rate=D,
                    activation='relu',
                    padding='same', # if P else 'valid',
                    name=label,
                    use_bias=True
                ))
            else:
                self.model.add(Conv2DTranspose(
                    filters=C,
                    kernel_size=(K, K),
                    strides=int(1 / S),
                    dilation_rate=D,
                    activation='relu',
                    padding='same', # if P else 'valid',
                    name=label,
                    use_bias=True
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


    def plot_image(self, orig_img):
        img = cv2.cvtColor(orig_img, cv2.COLOR_Lab2BGR)
        cv2.imshow('here is your fuck*ng image', img)
        cv2.waitKey(0)
        # img = cv2.cvtColor(orig_img, cv2.COLOR_Lab2RGB)
        # plt.imshow(img / 255.0)

m = CNN()

