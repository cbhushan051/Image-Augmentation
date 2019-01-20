# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 22:18:40 2019

@author: Shashi
"""

import cv2
import numpy as np

img = cv2.imread('mammo.png',0)
kernel = np.ones((5,5),np.uint8)

erosion = cv2.erode(img,kernel,iterations = 1)
cv2.imwrite('erosion_mammo.png',erosion)

dilation = cv2.dilate(img,kernel,iterations = 1)
cv2.imwrite('dilation_mammo.png',dilation)

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imwrite('opening_mammo.png',opening)

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('closing_mammo.png',closing)

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imwrite('gradient_mammo.png',gradient)

tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imwrite('tophat_mammo.png',tophat)

blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imwrite('blackhat_mammo.png',blackhat)