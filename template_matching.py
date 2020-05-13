import cv2 as cv
import numpy as np

image = cv.imread('catfamily.jpg')
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
# cv2.imread(path, flag)
# Parameters:
# path: A string representing the path of the image to be read.
# flag: It specifies the way in which image should be read. It’s default value is cv2.IMREAD_COLOR
""" All three types of flags are described below:

    1- cv2.IMREAD_COLOR: It specifies to load a color image. Any transparency of 
       image will be neglected. It is the default flag. Alternatively, we can pass
       integer value 1 for this flag.
    2- cv2.IMREAD_GRAYSCALE: It specifies to load an image in grayscale mode.
       Alternatively, we can pass integer value 0 for this flag.
    3- cv2.IMREAD_UNCHANGED: It specifies to load an image as such including
       alpha channel. Alternatively, we can pass integer value -1 for this flag.
    """
temp = cv.imread('cat.jpg',0)
width, hight = temp.shape[::-1]

result =cv.matchTemplate(gray,temp, cv.TM_CCOEFF_NORMED)
threshold = 0.7

location = np.where(result >= threshold)

for point in zip(*location[::-1]):
    cv.rectangle(image, point,(point[0]+width,point[1]+hight),(255,0,255),2)

cv.imshow('image', image)
cv.waitKey(0)
cv.destroyAllWindows()
