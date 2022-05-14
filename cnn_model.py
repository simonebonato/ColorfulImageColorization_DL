import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import tensorflow as tf

import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, InputLayer, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow import GradientTape
from custom_adam import AdamWeightDecayOptimizer
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from loss_function import *
from image_generator import *


# Customize how fit() method runs
class Custom_Seq(Sequential):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y_true = data

        y_true = soft_encoding2(image_ab=y_true, nn_finder=nn_finder, nb_q=nb_q)
        # y_true = v2(y_true)
        # y_true = tf.convert_to_tensor(y_true)

        with GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y_true = data

        y_true = soft_encoding2(image_ab=y_true, nn_finder=nn_finder, nb_q=nb_q)
        # y_true = v2(y_true)
        # y_true = tf.convert_to_tensor(y_true)

        y_pred = self(x, training=False)  # Forward pass

        # Compute the loss value
        # (the loss function is configured in `compile()`)
        self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class CNN:
    def __init__(self, input_shape=(256, 256), batch_size=1, init_lr=3e-5):
        '''
        input_shape: tuple indicating the desired shape of the input
        batch_size: number of samples for each batch of training
        '''
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.init_lr = init_lr

    def get_model(self):

        self.model = Custom_Seq()
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

        lr = ExponentialDecay(initial_learning_rate=self.init_lr, decay_steps=40623, decay_rate=0.8)
        # adam_weight = AdamWeightDecayOptimizer(beta_1=0.9, beta_2=0.99, learning_rate=lr, weight_decay_rate=10 ** -3)
        adam = Adam(beta_1=0.9, beta_2=0.99, learning_rate=lr)
        self.model.compile(loss=L_cl2, optimizer=adam, run_eagerly=True)
        print(self.model.summary())

        return self.model
