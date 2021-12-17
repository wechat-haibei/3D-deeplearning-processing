import numpy as np
import cv2 as cv


# 设置putText函数字体
font = cv.FONT_HERSHEY_SIMPLEX


def find_squares(img):
    squares = []
    img = cv.GaussianBlur(img, (7, 7), 0)
    img = cv.bilateralFilter(img,3,7,7)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bin = cv.Canny(gray, 150, 100, apertureSize=3)
    contours, _hierarchy = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    index = 0
    # 轮廓遍历
    sum = 0
    for cnt in contours:
        cnt_len = cv.arcLength(cnt, True)  # 计算轮廓周长
        cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)  # 多边形逼近
        # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
        if len(cnt) >= 4 and cv.contourArea(cnt)<1000 and cv.contourArea(cnt) > 50 and cv.isContourConvex(cnt):
            sum += cv.contourArea(cnt)
            M = cv.moments(cnt)  # 计算轮廓的矩
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])  # 轮廓重心

            cnt = cnt.reshape(-1, 2)
            if True:
                index = index + 1
                cv.putText(img, ("#%d" % index), (cx, cy), font, 0.7, (255, 0, 255), 2)
                squares.append(cnt)
    print("窗户数量为：",index)
    sum = sum/index
    sum = 9.72009090909091 * 9.72009090909091 * sum
    print("窗户面积为：",sum)
    return squares, img

def main():
    img = cv.imread("24_small.png")
    squares, img = find_squares(img)
    cv.drawContours(img, squares, -1, (0, 0, 255), 2)
    cv.imshow('squares', img)
    ch = cv.waitKey()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()