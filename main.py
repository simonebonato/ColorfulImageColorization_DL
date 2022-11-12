import matplotlib.pyplot as plt
import numpy as np

from Classprob_to_pointestimates import *
from cnn_model import *


@tf.autograph.experimental.do_not_convert
def main():
    input_shape = (256, 256)
    batch_size = 10
    train_path = 'data/train'
    val_path = 'data/val'
    saver_name = 'LessLayers'

    partition = {'train': (get_partitions(train_path, val_path))[0],
                 'val': (get_partitions(train_path, val_path))[1]}

    params = {'dim': input_shape,
              'batch_size': batch_size,
              'n_channels': (1, 2),
              'shuffle': True}

    training_generator = DataGenerator(partition['train'], **params)
    validation_generator = DataGenerator(partition['val'], **params)

    # Check if a model is already stored
    for file in os.listdir():
        if file == saver_name:
            model_exists = True
            break
        else:
            model_exists = False

    if model_exists:
        custom_objects = {'Custom_Seq': Custom_Seq, 'L_cl2': L_cl2, 'soft_encoding2': soft_encoding2, 'v2': v2}
        model = tf.keras.models.load_model(filepath=saver_name, custom_objects=custom_objects, compile=False)

        lr = ExponentialDecay(initial_learning_rate=3e-5, decay_steps=40623, decay_rate=0.8)
        adam = Adam(beta_1=0.9, beta_2=0.99, learning_rate=lr, clipvalue=5)
        model.compile(loss=L_cl2, optimizer=adam, run_eagerly=True)
        initial_epoch = 9
    else:
        model = CNN(input_shape, batch_size).get_model()
        initial_epoch = 0

    model_saver = ModelCheckpoint(filepath=saver_name, monitor='val_loss',
                                  save_best_only=True, mode='min')

    model.fit(
        x=training_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[model_saver],
        workers=1,
        use_multiprocessing=False,
        initial_epoch=initial_epoch)

    # Check quality of images
    test = training_generator.__getitem__(0)
    test_X, test_Y = test[0], test[1]
    y_pred = model.predict(test_X)
    y_pred = tf.nn.softmax(y_pred)

    gt_images = reconstruct_gt_image(X=test_X, y_true=test_Y)
    images = reconstruct_image(X=test_X, y_pred=y_pred)

    positions = [1, 2]
    gt_positions = []
    for i in range(gt_images.shape[0]):
        gt_pos = np.random.choice(positions)
        fake_pos = 1 if gt_pos == 2 else 2
        plt.subplot(1, 2, gt_pos)
        # plt.title('Ground Truth Image')
        plot_image_from_Lab(img=gt_images[i], gt=True)
        plt.subplot(1, 2, fake_pos)
        # plt.title('Model Image')
        plot_image_from_Lab(img=images[i], savename=f'{saver_name}/imgs/pic_{i}_{gt_pos}.png')


        gt_positions.append(gt_pos)



if __name__ == '__main__':
    main()
