# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 01:05:20 2019

@author: Shashi
"""

from keras.preprocessing import image

import tensorflow as tf
import numpy as np

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
 
test_image

test_image = image.img_to_array(test_image)

#Flipping left Right
flip_1 = np.fliplr(test_image)
imageflipped = image.array_to_img(flip_1)
imageflipped

#Flipping Up Down
flip_2 = np.flip(test_image, axis= 0)  #axis=0, x-axis, axis=1, y-axis
imageflipped2 = image.array_to_img(flip_2)
imageflipped2

#Rotation
img_rotation= np.rot90(test_image, k=1, axes=(0,1))
img_rot90 = image.array_to_img(img_rotation)
img_rot90

