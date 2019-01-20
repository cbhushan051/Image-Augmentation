# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 14:53:41 2019

@author: user
"""

'''Scaling an Image and Resizing :-'''
import cv2 
import numpy as np 

FILE_NAME = 'mammo.png'
FILE_NAME
try: 
	# Read image from disk. 
	img = cv2.imread(FILE_NAME) 

	# Get number of pixel horizontally and vertically. 
	(height, width) = img.shape[:2] 

	# Specify the size of image along with interploation methods. 
	# cv2.INTER_AREA is used for shrinking, whereas cv2.INTER_CUBIC 
	# is used for zooming. 
	res = cv2.resize(img, (int(width*0.7), int(height*0.9)), interpolation = cv2.INTER_CUBIC)
    #res_crop=cv2.c

	# Write image back to disk. 
	cv2.imwrite('result_Resizing2.jpg', res) 

except IOError: 
	print ('Error while reading files !!!') 

###
    
import cv2 
import numpy as np 

FILE_NAME = 'tumor.png'
FILE_NAME
try: 
	# Read image from disk. 
	img = cv2.imread(FILE_NAME) 

	# Get number of pixel horizontally and vertically. 
	(height, width) = img.shape[:2] 

	# Specify the size of image along with interploation methods. 
	# cv2.INTER_AREA is used for shrinking, whereas cv2.INTER_CUBIC 
	# is used for zooming. 
	res = cv2.resize(img, (int(width * 1.1), int(height )), interpolation = cv2.INTER_CUBIC) 

	# Write image back to disk. 
	cv2.imwrite('result_Scal.jpg', res) 

except IOError: 
	print ('Error while reading files !!!') 

'''Translating an Image :-
Translating an image means shifting it within a given frame of reference.'''
import cv2 
import numpy as np 

FILE_NAME = 'mammo.png'
# Create translation matrix. 
# If the shift is (x, y) then matrix would be 
# M = [1 0 x] 
#	 [0 1 y] 
# Let's shift by (100, 50). 
M = np.float32([[1, 0, -50], [0, 1, -20]]) 

try: 

	# Read image from disk. 
	img = cv2.imread(FILE_NAME) 
	(rows, cols) = img.shape[:2] 

	# warpAffine does appropriate shifting given the 
	# translation matrix. 
	res = cv2.warpAffine(img, M, (cols, rows)) 

	# Write image back to disk. 
	cv2.imwrite('translation2.png', res) 

except IOError: 
	print ('Error while reading files !!!') 


'''Edge detection in an Image :-'''

import cv2 
import numpy as np 

FILE_NAME = 'mammo.png'
try: 
	# Read image from disk. 
	img = cv2.imread(FILE_NAME) 

	# Canny edge detection. 
	edges = cv2.Canny(img, 100, 200) 

	# Write image back to disk. 
	cv2.imwrite('result.jpg', edges) 
except IOError: 
	print ('Error while reading files !!!') 


'''Image Gradients'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('mammo.png',0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()



#
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('mammo.png',0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

cv2.cvtColor(laplacian, 'lap.png', cv2.COLOR_GRAY2BGR)


backtorgb = cv2.cvtColor(laplacian,cv2.COLOR_GRAY2RGB)

laplacian_img = image.array_to_img(laplacian)
sobelx_img = image.array_to_img(sobelx)
sobely_img = image.array_to_img(sobely)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()

#Cropping
from keras.preprocessing import image

im = image.load_img('mammo.png')#.convert('L')
im = im.crop((33, 80, 250, 250))
im.save('crop2.png')

#RGB_to_Gray

backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)















