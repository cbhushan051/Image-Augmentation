# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 08:24:54 2019

@author: Shashi
"""

importmatplotlib.pyplotasplt
importnumpyasnp
from scipy import misc, ndimage
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
defplots(ims,figsize=(12,6),rows=1,interp=False,titles=None):
iftype(ims[0])isnp.ndarray:
ims=np.array(ims).astype(np.uint8)
if(ims.shape[-1]!=3):
ims=ims.transpose((0,2,3,1))
f=plt.figure(figsize=figsize)
cols=len(ims)//rowsiflen(ims)%2==0elselen(ims)//rows+1
foriinrange(len(ims)):
sp=f.add_subplot(rows,cols,i+1)
sp.axis('Off')
iftitlesisnotNone:
sp.set_title(titles[i],fontsize=16)
plt.imshow(ims[i],interpolation=Noneifinterpelse'none')
#Rotation
gen1=ImageDataGenerator(rotation_range=90)
#Width Shift
gen2=ImageDataGenerator(width_shift_range=0.1)
#Height_ Shift
gen3=ImageDataGenerator(height_shift_range=0.1)
gen4=ImageDataGenerator(shear_range=0.25)
gen5=ImageDataGenerator(zoom_range=0.3)
gen6=ImageDataGenerator(channel_shift_range=30.)
gen7=ImageDataGenerator(horizontal_flip=True)
image_path='dataset/single_prediction/cat_or_dog_1.jpg'
image=np.expand_dims(ndimage.imread(image_path),0)
plt.imshow(image[0])
aug_iter1=gen1.flow(image)
aug_images=[next(aug_iter1)[0].astype(np.uint8)foriinrange(1)]
plots(aug_images,figsize=(14,4),rows=1)
aug_iter2=gen2.flow(image)
aug_images=[next(aug_iter2)[0].astype(np.uint8)foriinrange(1)]
plots(aug_images,figsize=(14,4),rows=1)
aug_iter3=gen3.flow(image)
aug_images=[next(aug_iter3)[0].astype(np.uint8)foriinrange(1)]
plots(aug_images,figsize=(14,4),rows=1)
aug_iter4=gen4.flow(image)
aug_images=[next(aug_iter4)[0].astype(np.uint8)foriinrange(1)]
plots(aug_images,figsize=(14,4),rows=1)
aug_iter5=gen5.flow(image)
aug_images=[next(aug_iter5)[0].astype(np.uint8)foriinrange(1)]
plots(aug_images,figsize=(14,4),rows=1)
aug_iter6=gen6.flow(image)
aug_images=[next(aug_iter6)[0].astype(np.uint8)foriinrange(1)]
plots(aug_images,figsize=(14,4),rows=1)
aug_iter7=gen7.flow(image)
aug_images=[next(aug_iter7)[0].astype(np.uint8)foriinrange(1)]
plots(aug_images,figsize=(14,4),rows=1)