# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 22:58:29 2019

@author: Shashi
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('opencv_logo.png')

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()


#BLurring
blur = cv2.GaussianBlur(img,(5,5),0)

#Median Filtering
median = cv2.medianBlur(img,5)

#Median Filtering
blur = cv2.bilateralFilter(img,9,75,75)