# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:45:13 2019

@author: Shashi
"""

import cv2 as cv
import numpy as np
img = cv.imread('mammo.png',0)
kernel = np.ones((2,2),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)
cv.imwrite('morphological_segmentation_5iteration.png', erosion)