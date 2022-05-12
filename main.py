import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, InputLayer, UpSampling2D
from tensorflow.keras.models import Sequential
from cnn_model import CNN
from custom_adam import AdamWeightDecayOptimizer
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint
from loss_function import *
from image_generator import *
from cnn_model import *
from Classprob_to_pointestimates import  *


@tf.autograph.experimental.do_not_convert
def main():
    input_shape = (256, 256)
    batch_size = 8
    train_path = 'data/train'
    val_path = 'data/val'

    load_model = True

    partition = {'train': (get_partitions(train_path, val_path))[0],
                 'val': (get_partitions(train_path, val_path))[1]}
    params = {'dim': input_shape,
              'batch_size': batch_size,
              'n_channels': (1, 2),
              'shuffle': False}
    training_generator = DataGenerator(partition['train'], **params)
    validation_generator = DataGenerator(partition['val'], **params)

    if load_model:
        model = tf.keras.models.load_model(r'Best_Model')
    else:
        model = CNN(input_shape, batch_size).get_model()
        model_saver = ModelCheckpoint(filepath='Best_Model', monitor='val_loss',
                                      save_best_only=True, mode='min')
        model.fit(
            x=training_generator,
            epochs=20,
            validation_data=validation_generator,
            callbacks=[model_saver],
            workers=1,
            use_multiprocessing=False)


    test = training_generator.__getitem__(0)
    test_X, test_Y = test[0], test[1]
    y_pred = model.predict(test_X)
    rec = reconstruct_image(test_X, y_pred)
    plot_image_from_Lab(rec[0])


if __name__ == '__main__':
    main()