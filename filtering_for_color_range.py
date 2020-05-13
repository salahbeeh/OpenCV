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


    # now if we wanna get rid of the noise in the background
    # let's try smoothing with averaging
    kernel   = np.ones((15,15), np.float32)/225
    smoothed = cv.filter2D(result, -1, kernel)
    cv.imshow('smoothed', smoothed)

    # let's try gaussian bluring
    blur = cv.GaussianBlur(result,(15,15),0)
    cv.imshow('gaussian', blur)

    # let's try median
    median = cv.medianBlur(result,15)
    cv.imshow('median', median)

    #cv.imshow('mask',mask)
    cv.imshow('result', result)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

video.release()
cv.destroyAllWindows()
