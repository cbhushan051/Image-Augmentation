# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:43:28 2019

@author: Shashi
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('mammo.png',0)
ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

cv.imwrite('thresolding_binary.png', thresh1)
cv.imwrite('thresolding_binary_inverse.png', thresh2)
cv.imwrite('thresolding_trunc.png', thresh3)
cv.imwrite('thresolding_torezo.png', thresh4)
cv.imwrite('thresolding_torezo_inverse.png', thresh5)