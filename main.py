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

@tf.autograph.experimental.do_not_convert
def main():
    input_shape = (256, 256)
    batch_size = 2
    train_path = 'data/train'
    val_path = 'data/val'
    
    
    partition = {'train': (get_partitions(train_path, val_path))[0],
                    'val': (get_partitions(train_path, val_path))[1]}
    params = {'dim': input_shape,
                'batch_size': batch_size,
                'n_channels': (1, 2),
                'shuffle': False}
    training_generator = DataGenerator(partition['train'], **params)
    validation_generator = DataGenerator(partition['val'], **params)



    model = CNN(input_shape, batch_size).get_model()
    model_saver = ModelCheckpoint(filepath='Best_Model', monitor='val_loss',
                                      save_best_only=True, mode='min')
    model.fit(
            x = training_generator, 
            epochs = 5,
            validation_data = validation_generator, 
            callbacks=[model_saver])



if __name__ == '__main__':
    main()