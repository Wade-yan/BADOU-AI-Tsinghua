"""
@author: Wade Yan
do:彩色图像的灰度化，二值化
"""
#引用库
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.color import rgb2gray

#灰度化
img = cv2.imread('picts/lenna.png')
h,w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)

for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)

print(img_gray)
print('show grey:%s' % img_gray)
#cv2.imshow('show', img_gray)

plt.subplot(221)
img = plt.imread("picts/lenna.png")
plt.imshow(img)
print("---image lenna----")
print(img)

# 灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

#二值化
img_bin = np.where(img_gray >= 0.5, 1, 0)
print('---img-bin---')
print(img_bin)
print(img_bin.shape)

plt.subplot(223)
plt.imshow(img_bin, cmap='gray')
plt.show()
#cv2.waitKey(0)