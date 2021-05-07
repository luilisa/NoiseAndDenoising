import numpy as np
import matplotlib.pyplot as plt

from skimage import color, data, restoration
from skimage import io
from scipy.signal.signaltools import wiener
img = io.imread('Lena.jpg')
lena = color.rgb2gray(img)

PEAK = 3000


noise = np.random.poisson(lena / 255.0 * PEAK) / PEAK * 255
noisyImage = lena + noise
filtered_img = wiener(noisyImage, (5,5), noise = True)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))

plt.gray()
ax[0].imshow(lena)
ax[0].axis('off')
ax[0].set_title('Оригинал')

ax[1].imshow(noisyImage)
ax[1].axis('off')
ax[1].set_title('Пуассоновский шум')

ax[2].imshow(filtered_img)
ax[2].axis('off')
ax[2].set_title('Фильтр Винера')

fig.subplots_adjust(wspace=0.02, hspace=0.2,
                    top=0.9, bottom=0.05, left=0, right=1)

plt.show()