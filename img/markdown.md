题目要求：结合计算机视觉课程知识，识别并计算出楼层数目、最高高度、最宽宽度、最长长度、窗户数目、窗户面积。

### 1、计算楼层最高高度、最宽宽度、最长长度

##### 1）选取参照物（本例中选择大门作为参照物），拍摄人站在参照物前的图像，根据人与参照物在图像中的比例，计算出参照物的真实尺寸（如大门的实际高度）。

加载图片，转成灰度图

```
image = cv2.imread("353.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

用Sobel算子计算x，y方向上的梯度，之后在x方向上减去y方向上的梯度，通过这个减法，我们留下具有高水平梯度和低垂直梯度的图像区域。

```
gradX = cv2.Sobel(gray, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.cv.CV_32F, dx=0, dy=1, ksize=-1)

gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
```

去除图像上的噪声。首先使用低通滤泼器平滑图像（9 x 9内核）,这将有助于平滑图像中的高频噪声。低通滤波器的目标是降低图像的变化率。如将每个像素替换为该像素周围像素的均值。这样就可以平滑并替代那些强度变化明显的区域。

然后，对模糊图像二值化。梯度图像中不大于90的任何像素都设置为0（黑色）。 否则，像素设置为255（白色）。

```
# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)
```

利用cv2.findContours()函数取得图像的轮廓。cv2.findContours()函数返回两个值，一个是轮廓本身，还有一个是每条轮廓对应的属性。使用cv2.minAreaRect()函数求得包含点集最小面积的矩形，这个矩形是可以有偏转角度的，可以与图像的边界不平行。

```
(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.cv.BoxPoints(rect))

# draw a bounding box arounded the detected barcode and display the image
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.imshow("Image", image)
cv2.imwrite("contoursImage2.jpg", image)
cv2.waitKey(0)
```

找出最上端、最下端、最左端、最右端的点，并以此计算门的长、宽：

```
Xs = [i[0] for i in box]
Ys = [i[1] for i in box]
x1 = min(Xs)
x2 = max(Xs)
y1 = min(Ys)
y2 = max(Ys)
hight = y2 - y1
width = x2 - x1
```

![pp_withdoor](C:\Users\Administrator\Desktop\pp_withdoor.png)

![QQ图片20211217162158](C:\Users\Administrator\Desktop\QQ图片20211217162158.png)

用同样的方法计算人在图像中的尺寸

![pp24](C:\Users\Administrator\Desktop\pp24.png)

![pp_height](C:\Users\Administrator\Desktop\pp_height.png)

已知人的实际高度是170cm，计算人的实际高度/人的图像内高度，再用大门的图像内高度乘以这个比例，得到大门在实际中的高度。

```
per = 170/hight2
real_door = per * hight1
print(real_door)
```

得到门的实际高度（单位：cm）：

![QQ图片20211217162445](C:\Users\Administrator\Desktop\QQ图片20211217162445.png)

##### 2）根据建筑物整体的图像，计算参照物与建筑本身在那张图像上的高度和宽度，并根据参照物的实际高度与图像中高度的比例，求得建筑物的真实高度与宽度。

代码同上，得到结果如下：、

![QQ图片20211217163253](C:\Users\Administrator\Desktop\QQ图片20211217163253.png)

![QQ图片20211217163235](C:\Users\Administrator\Desktop\QQ图片20211217163235.png)

得到二者的尺寸分别为（单位：cm）：

![QQ图片20211217163920](C:\Users\Administrator\Desktop\QQ图片20211217163920.png)

求得参照物实际高度（427.68cm）与图像上的高度的比值，分别用楼房在图像中的高度与宽度乘以这个比值，得到楼房的实际尺寸。

```
per = 427.68 / height1
real_height = per * height2
real_width = per * height2
print("楼房高度：",real_height,"楼房宽度",real_width)
```

得到楼房的真实尺寸（单位：cm）：

![QQ图片20211217164520](C:\Users\Administrator\Desktop\QQ图片20211217164520.png)

##### 楼房的长度、8号楼以及22号楼的尺寸可以用同样的方法得出，在此省略。



### 2、识别并计算窗户的数量、面积

```
import numpy as np
import cv2 as cv

font = cv.FONT_HERSHEY_SIMPLEX
```

定义识别函数，并以如下步骤实现要求的功能：

1、分别对图像进行高斯平滑、双边平滑处理图像

2、将图像转化成灰度图

3、得到轮廓，并使用cv.findContours()函数得到轮廓的几何形状

4、遍历所有轮廓，利用：

①边数是否大于等于4条

②面积是否大于50

③形状是否为凸的

三个条件识别轮廓中的矩形并计数

5、计算矩形的总面积，并计算矩形的平均面积

6、利用计算建筑比例时计算出的，同一张图中的，门的实际高度与图中高度的比例，得到窗户的实际面积

```
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
    sum = 9.72 * 9.72 * sum
    print("窗户面积为：",sum)
    return squares, img
```

识别窗户的结果：

![QQ图片20211217172052](C:\Users\Administrator\Desktop\QQ图片20211217172052.png)

得到的计数结果、窗户面积为（单位分别为：个、平方厘米）：

![QQ图片20211217175737](C:\Users\Administrator\Desktop\QQ图片20211217175737.png)

### 3、计算楼层的层数

```
import cv2 as cv
```

首先将图像转化为灰度图，再转化为二进制图像

```
def ToBinray():
    global imgray, binary
    # 1、灰度图
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('imgray', imgray)

    # 2、二进制图像
    ret, binary = cv.threshold(imgray, 127, 255, 0)
    # 阈值 二进制图像
    cv.imshow('binary', binary)
```

输出的灰度图为：

![QQ图片20211217180634](C:\Users\Administrator\Desktop\QQ图片20211217180634.png)

输出的二进制图为

![QQ图片20211217180945](C:\Users\Administrator\Desktop\QQ图片20211217180945.png)

提取轮廓

```
def GetGontours():
    # 1、根据二值图找到轮廓
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # 轮廓      层级                               轮廓检索模式(推荐此)  轮廓逼近方法

    # 2、画出轮廓
    dst = cv.drawContours(img, contours, -1, (0, 0, 255), 3)
    #                           轮廓     第几个(默认-1：所有)   颜色       线条厚度

    cv.imshow('dst', dst)
```

结果为

![QQ图片20211217181048](C:\Users\Administrator\Desktop\QQ图片20211217181048.png)

```
if __name__ == '__main__':
    img = cv.imread('24_2.png')
    cv.imshow('img', img)

    ToBinray()  # 转二进制

    GetGontours()  # 提取轮廓

    cv.waitKey(0)
```

依据同上一步中的，识别窗户的原理，计算出楼层数目为：

![QQ图片20211217181338](C:\Users\Administrator\Desktop\QQ图片20211217181338.png)

8号楼、22号楼的楼层数目方法同上。

