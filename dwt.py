import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data
from keras.preprocessing import image


# Load image
#original = pywt.data.camera()

imge = image.load_img('mammo.png', grayscale = True, target_size = (255, 281))
original = image.img_to_array(imge)

origi = original.transpose()
origin = np.reshape(origi, (281,255))
origin = origin.transpose()

original = origin

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()