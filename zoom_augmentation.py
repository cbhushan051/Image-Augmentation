# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 22:46:26 2019

@author: Shashi
"""

import numpy as np
from scipy.ndimage.interpolation import zoom
from PIL import Image
from keras.preprocessing import image
test_image = image.load_img('mammo.png', target_size=(255, 281))

test_image = image.img_to_array(test_image)
#import Image 
zoom_factor = 0.05 # 5% of the original image 
img = Image.open('mammo.png')
#image_array =np.img
zoomed_img = zoom.clipped_zoom(image_array, zoom_factor)
misc.imsave('zoomed1.png', zoomed_img)

#Method 2

def loadImageData(img, distort = False):
    c, fn = img
    img = scipy.ndimage.imread(fn, True)

    if distort:
        img = scipy.ndimage.zoom(img, 1 + 0.05 * rnd(), mode = 'constant')
        img = scipy.ndimage.rotate(img, 10 * rnd(), mode = 'constant')
        print(img.shape)

    img = img - np.min(img)
    img = img / np.max(img)
    img = np.reshape(img, (1, *img.shape))

    y = np.zeros(ncats)
    y[c] = 1
    return (img, y)
loadImageData('mammo.png')

#Method 2


import numpy as np
from scipy.ndimage.interpolation import zoom
from PIL import Image
from keras.preprocessing import image

Z_img = image.load_img('mammo.png',target_size=(255, 281))
Zoom= image.random_zoom(Z_img, )
Z_img

# Method 3

import numpy as np
from scipy.ndimage import zoom


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

imge = image.load_img('mammo.png', grayscale = True, target_size = (255, 281))
original = image.img_to_array(imge)

origi = original.transpose()
origin = np.reshape(origi, (281,255))
origin = origin.transpose()

Z_img = clipped_zoom(origin, zoom_factor=20)

import cv2
backtorgb = cv2.cvtColor(Z_img,cv2.COLOR_GRAY2RGB)
Zoom_img = image.array_to_img(backtorgb)
Zoom_img








 