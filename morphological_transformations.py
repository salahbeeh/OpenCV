import cv2 as cv
import numpy as np


video = cv.VideoCapture(-1)

while True:
    _ , frame = video.read()
    hsv = cv.cvtColor(frame ,cv.COLOR_BGR2HSV)

    # hsv hue sat Value
    lower  = np.array([130,0,0])
    higher = np.array([200,255,255])

    mask   = cv.inRange(hsv,lower,higher)
    result = cv.bitwise_and(frame,frame,mask =mask)

    kernel  = np.ones((5,5),np.uint8)
    erosion = cv.erode(mask,kernel,iterations = 1)
    dilation = cv.dilate(mask,kernel,iterations = 1)

    # opening removes the false positives which are the noise in the background
    opening = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel)
    # closing removes the false negatives which are mistaken detectied in the object we tring to filter
    closing = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel)

    cv.imshow('erosion', erosion)
    cv.imshow('dilation', dilation)
    cv.imshow('opening', opening)
    cv.imshow('closing', closing)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

video.release()
cv.destroyAllWindows()
