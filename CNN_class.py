from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization

class CNN:
    def __init__(self):
        print('-- Creating CNN model --')

        model = Sequential()

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
            (56, 128, 1, 2, 4, 4, False, True),

        )

        for (X, C, S, D, Sa, De, BN, L) in conv_layers:
            model.add(Conv2D(
                filters=C,
                kernel_size=X,
                strides=S,
                dilation_rate=D,
                activation='relu'
            ))

            if BN:
                model.add(BatchNormalization())

    def train_val_split(self, train_p):
        pass

    def fit(self, X_train, Y_train, X_val, Y_val):
        pass

    def predict(self, X_test):
        pass

    def loss_function(self):
        pass







