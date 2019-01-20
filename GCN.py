# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 23:31:50 2019

@author: Shashi
"""

import numpy
import scipy
import scipy.misc
from PIL import Image


def global_contrast_normalization(filename, s, lmda, epsilon):
    X = numpy.array(Image.open(filename))

    # replacement for the loop
    X_average = numpy.mean(X)
    print('Mean: ', X_average)
    X = X - X_average

    # `su` is here the mean, instead of the sum
    contrast = numpy.sqrt(lmda + numpy.mean(X**2))

    X = s * X / max(contrast, epsilon)

    # scipy can handle it
    scipy.misc.imsave('GCN_mammo.png', X)


global_contrast_normalization("mammo.png", 1, 10, 0.000000001)