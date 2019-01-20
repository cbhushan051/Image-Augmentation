# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 21:40:28 2019

@author: Shashi
"""

from skimage import io
from skimage import transform as tf

# Load the image as a matrix
image = io.imread("mammo.png")

# Create Afine transform
afine_tf = tf.AffineTransform(shear=-0.2)

# Apply transform to image data
modified = tf.warp(image, inverse_map=afine_tf)

# Display the result
io.imshow(modified)
io.show()