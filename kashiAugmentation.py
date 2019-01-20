import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

def plots(ims, figsize=(12,6),rows=1,interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims=np.array(ims).astype(np.uint8)
        if (ims.shape[-1] !=3):
            ims=ims.transpose((0,2,3,1))
    f=plt.figure(figsize=figsize)        
    cols=len(ims)//rows if len(ims)%2==0 else len(ims)//rows+1
    for i in range (len(ims)):
        sp=f.add_subplot(rows, cols,i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none') 

gen1=ImageDataGenerator(rotation_range=120) 
gen2=ImageDataGenerator(width_shift_range=0.5) 
gen3=ImageDataGenerator(height_shift_range=0.5) 
gen4=ImageDataGenerator(shear_range=0.5) 
gen5=ImageDataGenerator(zoom_range=0.5)
gen6=ImageDataGenerator(channel_shift_range=90.)
gen7=ImageDataGenerator(vertical_flip=True)   
gen8=ImageDataGenerator(horizontal_flip=True)   
gen9=ImageDataGenerator()
#image_path='dog1.jpg'

image=np.expand_dims(ndimage.imread('mammo.png'),0)
plt.imshow(image[0]) 

aug_iter1=gen1.flow(image)
aug_images=[next(aug_iter1)[0].astype(np.uint8) for i in range(1)]
plots(aug_images, figsize=(14,4),rows=1)


aug_iter2=gen2.flow(image)
aug_images=[next(aug_iter2)[0].astype(np.uint8) for i in range(1)]
plots(aug_images, figsize=(14,4),rows=1)


aug_iter3=gen3.flow(image)
aug_images=[next(aug_iter3)[0].astype(np.uint8) for i in range(1)]
plots(aug_images, figsize=(14,4),rows=1)

aug_iter4=gen4.flow(image)
aug_images=[next(aug_iter4)[0].astype(np.uint8) for i in range(1)]
plots(aug_images, figsize=(14,4),rows=1)

aug_iter5=gen5.flow(image)
aug_images=[next(aug_iter5)[0].astype(np.uint8) for i in range(1)]
plots(aug_images, figsize= (255,281),rows=1)



aug_iter6=gen6.flow(image)
aug_images=[next(aug_iter6)[0].astype(np.uint8) for i in range(1)]
plots(aug_images, figsize=(14,4),rows=1)

aug_iter7=gen7.flow(image)
aug_images=[next(aug_iter7)[0].astype(np.uint8) for i in range(1)]
plots(aug_images, figsize=(14,4),rows=1)


aug_iter8=gen8.flow(image)
aug_images=[next(aug_iter7)[0].astype(np.uint8) for i in range(1)]
plots(aug_images, figsize=(14,4),rows=1)






