from keras.models import Sequential
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import numpy as np
from keras.layers import Conv2D, BatchNormalization
from keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
      """Constructs a AdamWeightDecayOptimizer."""
      super(AdamWeightDecayOptimizer, self).__init__(False, name)

      self.learning_rate = learning_rate
      self.weight_decay_rate = weight_decay_rate
      self.beta_1 = beta_1
      self.beta_2 = beta_2
      self.epsilon = epsilon
      self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
      """See base class."""
      assignments = []
      for (grad, param) in grads_and_vars:
        if grad is None or param is None:
          continue

        param_name = self._get_variable_name(param.name)

        m = tf.get_variable(
            name=param_name + "/adam_m",
            shape=param.shape.as_list(),
            dtype=tf.float32,
            trainable=False,
            initializer=tf.zeros_initializer())
        v = tf.get_variable(
            name=param_name + "/adam_v",
            shape=param.shape.as_list(),
            dtype=tf.float32,
            trainable=False,
            initializer=tf.zeros_initializer())

        # Standard Adam update.
        next_m = (
            tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
        next_v = (
            tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                      tf.square(grad)))

        update = next_m / (tf.sqrt(next_v) + self.epsilon)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want ot decay the weights in a manner that doesn't interact
        # with the m/v parameters. This is equivalent to adding the square
        # of the weights to the loss with plain (non-momentum) SGD.
        if self._do_use_weight_decay(param_name):
          update += self.weight_decay_rate * param

        update_with_lr = self.learning_rate * update

        next_param = param - update_with_lr

        assignments.extend(
            [param.assign(next_param),
             m.assign(next_m),
             v.assign(next_v)])
      return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
      """Whether to use L2 weight decay for `param_name`."""
      if not self.weight_decay_rate:
        return False
      if self.exclude_from_weight_decay:
        for r in self.exclude_from_weight_decay:
          if re.search(r, param_name) is not None:
            return False
      return True

    def _get_variable_name(self, param_name):
      """Get the variable name from the tensor name."""
      m = re.match("^(.*):\\d+$", param_name)
      if m is not None:
        param_name = m.group(1)
      return param_name


class CNN:
    def __init__(self):
        print('-- Creating CNN model --')

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

            (56, 128, .5, 1, 4, 4, False, False),
            (56, 128, 1, 1, 4, 4, False, False),
            (56, 128, 1, 1, 4, 4, False, True),

        )

        for (X, C, S, D, Sa, De, BN, L) in conv_layers:
            # if S
            self.model.add(Conv2D(
                filters=C,
                kernel_size=X,
                strides=S,
                dilation_rate=D,
                activation='relu'
            ))

            if BN:
                self.model.add(BatchNormalization())

        adam = AdamWeightDecayOptimizer(beta_1=0.9, beta_2=0.99, learning_rate=3e-5, weight_decay_rate=10**-3)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam)

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
