# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 23:44:27 2019

@author: kashif Reza
"""
from keras.preprocessing import image
test_image = image.load_img('mammo.png', target_size=(255, 281))

test_image = image.img_to_array(test_image)

import tensorflow as tf
import numpy as np
# NumPy.'img' = A single image.
flip_1 = np.fliplr(test_image)
imageflipped = image.array_to_img(flip_1)
imageflipped

flip_2 = np.flipud(test_image)
imageflipped2 = image.array_to_img(flip_2)
imageflipped2
# TensorFlow. 'x' = A placeholder for an image.
shape = [64, 64, 3]
x = tf.placeholder(dtype = tf.float32, shape = shape)
flip_2 = tf.image.flip_up_down(test_image)
flip_3 = tf.image.flip_left_right(x)
flip_4 = tf.image.random_flip_up_down(x)
flip_5 = tf.image.random_flip_left_right(x)

'''test_image  =np.expand_dims(flip_1, axis = 0)


from matplotlib import pyplot

for X_batch in flip_1:
    
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(X_batch[i], cmap=pyplot.get_cmap('gray'))
    
        #Show the plot
        pyplot.show()'''
        
        
# Placeholders: 'x' = A single image, 'y' = A batch of images
# 'k' denotes the number of 90 degree anticlockwise rotations
#shape = [height, width, channels]
x = tf.placeholder(dtype = tf.float32, shape = shape)
rot_90 = tf.image.rot90(test_image, k=1)
tf.image.convert_image_dtype(flip_2, dtype=)

real_img =  image.array_to_img(test_image)
real_img

imgrotation= np.rot90(test_image, k=4)
img_rot90 = image.array_to_img(imgrotation)
img_rot90

#crop
img_scal= np.resize(test_image, (128,128,3))
img_scal2 = image.array_to_img(img_scal)
img_scal2

#
img_reshape= np.reshape(test_image, 128, 128, 3))
img_shape2 = image.array_to_img(img_reshape)
img_shape2

rot_180 = tf.image.rot90(img, k=2)
# To rotate in any angle. In the example below, 'angles' is in radians
shape = [batch, height, width, 3]
y = tf.placeholder(dtype = tf.float32, shape = shape)
rot_tf_180 = tf.contrib.image.rotate(y, angles=3.1415)
# Scikit-Image. 'angle' = Degrees. 'img' = Input Image
# For details about 'mode', checkout the interpolation section below.
rot = skimage.transform.rotate(img, angle=45, mode='reflect')


#Any angle Rotation

from scipy.ndimage import rotate
from scipy.misc import face
from matplotlib import pyplot as plt
from keras.preprocessing import image
test_image = image.load_img('mammo.png', target_size=(255, 281, 3))

test_image = image.img_to_array(test_image)


#img = face()
img1 = test_image
rot = rotate(img1, 60, reshape=False)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img1)
ax[1].imshow(rot)



