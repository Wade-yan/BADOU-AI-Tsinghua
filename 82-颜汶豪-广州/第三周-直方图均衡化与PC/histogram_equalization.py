#直方图均衡化
'''
步骤：
1、依次扫描原始灰度图像的每一个像素，计算出图像的灰度直方图
2、计算灰度图的累加直方图
3、根据累加直方图和直方图均衡化原理得到输入与输出的映射关系
4、根据映射关系得到结果
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt


def my_gray_hist_eq(img):
    '''
    img:a picture
    :return:
    '''

    #获取图像大小
    h,w = img.shape[:2]
    #申请一个图像空间
    dst = np.zeros((h,w),dtype=np.uint8)
    #图片转灰度直方图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    #直方图统计
    gray_hist = []
    for i in range(0,256):
        gray_hist.append(0)
    for i in range(h):
        for j in range(w):
            gray_hist[gray[i,j]] += 1
    #归一化
    for k in range(256):
        gray_hist[k] /= h*w
    #直方图累加
    for k in range(1,256):
        gray_hist[k] = gray_hist[k-1]+gray_hist[k]
    #均衡化
    for k in range(256):
        gray_hist[k] = (np.uint8)(255*gray_hist[k]+0.5)
    #赋值
    for i in range(h):
        for j in range(w):
            dst[i,j] = gray_hist[gray[i,j]]
    return dst


# 创建空列表(数组)
def createEmptyList(size):
    newList = []
    for eachNum in range(0, size):
        newList.append(0)
    return newList


# 创建空图像矩阵
def createEmptyImage(rows, cols, type):
    img = np.zeros((rows, cols), dtype=type)
    return img


# 直方图均衡化
def histequaLize(src, dst):
    # step 1 校验参数#
    assert (type(src) == np.ndarray)
    assert (src.dtype == np.uint8)
    assert (type(dst) == np.ndarray)
    assert (dst.dtype == np.uint8)

    # step 2 直方图统计#
    hist = createEmptyList(256)
    rows, cols = src.shape
    for r in range(rows):
        for c in range(cols):
            hist[src[r, c]] += 1

    # step 3 直方图归一化#
    for each in range(256):
        hist[each] /= rows * cols

    # step 4 直方图累加#
    for each in range(1, 256):
        hist[each] = hist[each - 1] + hist[each]

    # step 4 均衡#
    for each in range(256):
        hist[each] = (np.uint8)(255 * hist[each] + 0.5)

    for r in range(rows):
        for c in range(cols):
            dst[r, c] = hist[src[r, c]]


def histMain():
    img = cv2.imread('lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = createEmptyImage(img.shape[0], img.shape[1], np.uint8)
    histequaLize(gray, dst)
    cv2.imshow('histEnhance.jpg', dst)
    cv2.waitKey(0)


if __name__ == '__main__':
    #histMain()
    img = cv2.imread('lenna.png')
    # 打印原始图像
    cv2.imshow('origin', img)
    #打印转换后的图片
    cv2.imshow('change',my_gray_hist_eq(img))
    cv2.waitKey(0)