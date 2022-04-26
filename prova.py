from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import argparse

import cv2

# construct the argument parser and parse the arguments
filename = r'C:\Users\simon\PycharmProjects\DeepLearningPython\GroupProject\data\dataset'
orig = cv2.imread(filename)
print("[INFO] loading and preprocessing image...")
image = image_utils.load_img(filename, target_size=(224, 224))
image = image_utils.img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)