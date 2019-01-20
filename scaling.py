# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:51:01 2019

@author: Shashi
"""

import cv2
import numpy as np

img = cv2.imread('mammo.png')

res = cv2.resize(img,None,fx=0.9, fy=0.9, interpolation = cv2.INTER_CUBIC)

cv2.imwrite('scaled2.png', res)

#OR

height, width = img.shape[:2]
res = cv2.resize(img,(0.7*width, 0.7*height), interpolation = cv2.INTER_CUBIC)