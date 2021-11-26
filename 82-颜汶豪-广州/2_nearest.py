"""
@author: Wade Yan
do:最邻近算法
"""

import cv2
import numpy as np
#最邻近算法
def function(img):
    h,w,chns = img.shape   #从原图像中获取长边像素，短边像素，和像素通道
    empImg = np.zeros((800,800,chns),np.uint8)   #新建一个相同通道数的大小为800*800的图像
    nh = 800/h  #高度比例
    nw = 800/w  #宽度比例
    #在目标图像矩阵中，每一点的像素值通过比例运算后，四舍五入后得到一个整数值，该整数值代表原图像上的像素点
    for i in range(800):
        for j in range(800):
            x = int(i/nh)
            y = int(j/nw)
            empImg[i,j] = img[x,y]  #目标图像赋值
    return empImg

img = cv2.imread('picts/lenna.png')
c_img = function(img)
print(c_img)
print(c_img.shape)
cv2.imshow('img',img)
cv2.imshow('nearest interp',c_img)
cv2.waitKey(0)