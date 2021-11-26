"""
@author: Wade Yan
do:双线性插值算法
"""
import cv2
import numpy as np

#Start：双线性插值算法
def bilinear_interpolation(img,out_dim):
    src_h,src_w,chaanels = img.shape
    dst_h,dst_w = out_dim[1],out_dim[0]#为了符合人类输入习惯：长*宽
    #如果目标图像大小与原图像一致，则返回图片的备份，为什么是备份而不能直接返回呢
    if src_h==dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    scale_x,scale_y = float(src_w)/dst_w,float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
               #找对应原图像的原点，中心对称法定点
                src_x = (dst_x+0.5)*scale_x-0.5
                src_y = (dst_y+0.5)*scale_y-0.5
                #直接定点
                #src_x = dst_x*scale_x
                #src_y = dst_y*scale_y
                #找到即将用于插值的点的坐标，在原图上根据已有的两点坐标找邻近的两点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0+1,src_w-1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0+1,src_h-1)
                #代入公式，计算插值
                temp0 = (src_x1-src_x)*img[src_y0,src_x0,i] + (src_x-src_x0)*img[src_y0,src_x1,i]
                temp1 = (src_x1-src_x)*img[src_y1,src_x0,i] + (src_x-src_x0)*img[src_y1,src_x1,i]
                #目标图像赋值
                dst_img[dst_y,dst_x,i] = int((src_y1-src_y)*temp0 + (src_y-src_y0)*temp1)

    return dst_img
#End:双线性插值算法
img = cv2.imread('picts/lenna.png')
dst = bilinear_interpolation(img,(700,700))
cv2.imshow('bilinear interp',dst)
cv2.waitKey(0)

