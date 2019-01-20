# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 22:15:21 2019

@author: Shashi
"""

import numpy as np
import cv2

img = cv2.imread('mammo.png',0)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

cv2.imwrite('clahe_mammo.png',cl1)