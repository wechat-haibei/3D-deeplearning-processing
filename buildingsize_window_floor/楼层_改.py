# 参考：
# 计算窗户与楼层数目：
# https://cloud.tencent.com/developer/article/1675022
# 计算楼层高度：
# https://blog.csdn.net/liqiancao/article/details/55670749

# 轮廓提取
import cv2 as cv


# 转二进制图像
def ToBinray():
    global imgray, binary
    # 1、灰度图
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('imgray', imgray)

    # 2、二进制图像
    ret, binary = cv.threshold(imgray, 127, 255, 0)
    # 阈值 二进制图像
    cv.imshow('binary', binary)


# 提取轮廓
def GetGontours():
    # 1、根据二值图找到轮廓
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # 轮廓      层级                               轮廓检索模式(推荐此)  轮廓逼近方法

    # 2、画出轮廓
    dst = cv.drawContours(img, contours, -1, (0, 0, 255), 3)
    #                           轮廓     第几个(默认-1：所有)   颜色       线条厚度

    cv.imshow('dst', dst)


if __name__ == '__main__':
    img = cv.imread('24_2.png')
    cv.imshow('img', img)

    ToBinray()  # 转二进制

    GetGontours()  # 提取轮廓

    cv.waitKey(0)
