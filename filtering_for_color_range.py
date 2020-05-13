import cv2 as cv
import numpy as np


video = cv.VideoCapture(-1)

while True:
    _ , frame = video.read()
    hsv = cv.cvtColor(frame ,cv.COLOR_BGR2HSV)

    # hsv hue sat Value
    lower  = np.array([150,0,0])
    higher = np.array([220,255,255])

    mask   = cv.inRange(hsv,lower,higher)
    result = cv.bitwise_and(frame,frame,mask =mask)

    cv.imshow('mask',mask)
    cv.imshow('result', result)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

video.release()
cv.destroyAllWindows()
