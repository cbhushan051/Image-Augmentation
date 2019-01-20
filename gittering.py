import cv2
import numpy as np
#from pylab import *
import matplotlib.pyplot as plt
#import image

img = cv2.imread(r'mammo.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # cv2 defaul color code is BGR
h,w,c = img.shape # (768, 1024, 3)
noise = np.random.randint(0,4,(h, w)) # design jitter/noise here
zitter = np.zeros_like(img)
zitter[:,:,1] = noise
noise_added = cv2.add(img, zitter)
i = int(h/4)

# combined = np.vstack((img[:h,:,:], noise_added[h:,:,:]))
# combined = np.vstack((img[:h/2,:,:], noise_added[h/2:,:,:]))
# combined = np.vstack((img[:h/3,:,:], noise_added[h/3:,:,:]))

combined = np.vstack((img[:i,:,:], noise_added[i:,:,:]))
# imshow(combined, interpolation='none')

# this code is used to remove white frame in the pic
fig = plt.figure(frameon=False)
fig.set_size_inches(3.46,3.93)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(combined, aspect='auto')
# save the pic
fig.savefig(r'gitter2.png')