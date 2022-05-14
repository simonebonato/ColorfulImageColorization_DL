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
from Classprob_to_pointestimates import *


@tf.autograph.experimental.do_not_convert
def main():
    input_shape = (256, 256)
    batch_size = 16
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

    # Check if a model is already stored
    for file in os.listdir():
        if file == 'Best_Model':
            model_exists = True
            break
        else:
            model_exists = False

    if model_exists:
        custom_objects = {'Custom_Seq': Custom_Seq, 'L_cl2': L_cl2, 'soft_encoding2': soft_encoding2, 'v2': v2}
        model = tf.keras.models.load_model(filepath='Best_Model', custom_objects=custom_objects, compile=False)

        lr = ExponentialDecay(initial_learning_rate=3e-5, decay_steps=40623, decay_rate=0.8)
        adam = Adam(beta_1=0.9, beta_2=0.99, learning_rate=lr)
        model.compile(loss=L_cl2, optimizer=adam, run_eagerly=True)
        initial_epoch = 7
    else:
        model = CNN(input_shape, batch_size).get_model()
        initial_epoch = 0

    model_saver = ModelCheckpoint(filepath='Best_Model', monitor='val_loss',
                                  save_best_only=True, mode='min')
    model.fit(
        x=training_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[model_saver],
        workers=1,
        use_multiprocessing=False,
        initial_epoch=initial_epoch)

    # Check quality of one image
    # test = training_generator.__getitem__(0)
    # test_X, test_Y = test[0], test[1]
    # y_pred = model.predict(test_X)

    # images = reconstruct_image(X=test_X, y_pred=y_pred)
    # plot_image_from_Lab(img=images[0])


if __name__ == '__main__':
    main()
