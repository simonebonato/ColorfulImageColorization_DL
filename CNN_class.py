import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import tensorflow as tf

import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, InputLayer, UpSampling2D
from tensorflow.keras.models import Sequential
from adam_class import AdamWeightDecayOptimizer
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from Loss_func import *
from image_gen import *


class CNN:
    def __init__(self, input_shape=(256, 256), batch_size=4):
        '''
        input_shape: tuple indicating the desired shape of the input
        batch_size: number of samples for each batch of training
        '''
        self.input_shape = input_shape
        self.batch_size = batch_size

        self.get_generators()

        self.model = Sequential()
        self.model.add(InputLayer(
            input_shape=(*self.input_shape, 1),
            batch_size=self.batch_size,
            name='input',
        ))

        # see Table 4 - page 24
        # C : number of filters
        # S : stride
        # D : kernel dilation
        # BN : Batch norm (True or False)
        # Padding : True - same, False - valid
        # layer structure ('name', C, S, D, K, BN, activation)
        conv_layers = (
            ('conv1_1', 64, 1, 1, 3, False, 'relu'),
            ('conv1_2', 64, 2, 1, 3, True, 'relu'),

            ('conv2_1', 128, 1, 1, 3, False, 'relu'),
            ('conv2_2', 128, 2, 1, 3, True, 'relu'),

            ('conv3_1', 256, 1, 1, 3, False, 'relu'),
            ('conv3_2', 256, 1, 1, 3, False, 'relu'),
            ('conv3_3', 256, 2, 1, 3, True, 'relu'),

            ('conv4_1', 512, 1, 1, 3, False, 'relu'),
            ('conv4_2', 512, 1, 1, 3, False, 'relu'),
            ('conv4_3', 512, 1, 1, 3, True, 'relu'),

            ('conv5_1', 512, 1, 2, 3, False, 'relu'),
            ('conv5_2', 512, 1, 2, 3, False, 'relu'),
            ('conv5_3', 512, 1, 2, 3, True, 'relu'),

            ('conv6_1', 512, 1, 2, 3, False, 'relu'),
            ('conv6_2', 512, 1, 2, 3, False, 'relu'),
            ('conv6_3', 512, 1, 2, 3, True, 'relu'),

            ('conv7_1', 512, 1, 1, 3, False, 'relu'),
            ('conv7_2', 512, 1, 1, 3, False, 'relu'),
            ('conv7_3', 512, 1, 1, 3, True, 'relu'),

            ('conv8_1', 256, .5, 1, 4, False, 'relu'),
            ('conv8_2', 256, 1, 1, 3, False, 'relu'),
            ('conv8_3', 256, 1, 1, 3, False, 'relu'),

            ('conv_out', 313, 1, 1, 1, False, 'softmax')

        )
        """TODO: watch their structure: https://github.com/foamliu/Colorful-Image-Colorization/blob/master/model.py"""
        for (label, C, S, D, K, BN, activation) in conv_layers:
            if S >= 1:
                self.model.add(Conv2D(
                    filters=C,
                    kernel_size=(K, K),
                    strides=S,
                    dilation_rate=D,
                    activation=activation,
                    padding='same',  # if P else 'valid',
                    name=label,
                    use_bias=True
                ))
            else:
                self.model.add(UpSampling2D(size=(2, 2), name='upsample'))
                # self.model.add(Conv2DTranspose(
                #     filters=C,
                #     kernel_size=(K, K),
                #     strides=int(1 / S),
                #     dilation_rate=D,
                #     activation=activation,
                #     padding='same',  # if P else 'valid',
                #     name=label,
                #     use_bias=True
                # ))

            if BN:
                self.model.add(BatchNormalization(name=f'BN_{label[4]}'))

        """TODO: add the layers to the conv_ayers above"""
        """add these to get an output with dims [1, 256, 256, 2]"""
        # self.model.add(Softmax(axis=-1, name='softmax'))
        # self.model.add(Conv2D(
        #     filters=2,
        #     kernel_size=(1, 1),
        #     strides=1,
        #     dilation_rate=1,
        #     activation='relu',
        #     padding='same',  # if P else 'valid',
        #     name='dunno',
        #     use_bias=False
        # ))
        # self.model.add(UpSampling2D(size=(4, 4), interpolation='bilinear', name='upsample'))

        lr = ExponentialDecay(initial_learning_rate=3e-5, decay_steps=10, decay_rate=0.01)
        adam_weight = AdamWeightDecayOptimizer(beta_1=0.9, beta_2=0.99, learning_rate=lr, weight_decay_rate=10 ** -3)
        self.model.compile(loss=L_cl, optimizer=adam_weight, run_eagerly=True)
        print(self.model.summary())

        self.model.fit(x=self.training_generator,
                       epochs=5)

    def get_generators(self):
        train_path = 'data/train'
        val_path = 'data/val'
        partition = {'train': (get_partitions(train_path, val_path))[0],
                     'val': (get_partitions(train_path, val_path))[1]}

        params = {'dim': self.input_shape,
                  'batch_size': self.batch_size,
                  'n_channels': (1, 2),
                  'shuffle': False}

        self.training_generator = DataGenerator(partition['train'], **params)
        self.validation_generator = DataGenerator(partition['val'], **params)


if __name__ == '__main__':
    m = CNN()
