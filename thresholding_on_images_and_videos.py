import  cv2 as cv
import numpy as np

image = cv.imread('bookpage.jpg')

# frist threshold is the binary
# cv2.threshold( image , critical point, converted values, type of threshold)
# the critical point is where any thing above it will be converted to the (converted value)
retval, threshold = cv.threshold(image, 12, 255, cv.THRESH_BINARY)
cv.imshow('binary threshold', threshold)

# let's try convert the original image to grayscale to see maybe it might help
grayscale = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
retval, threshold2 = cv.threshold(grayscale , 12, 255, cv.THRESH_BINARY)
cv.imshow('grayscaled', threshold2)


# adaptive threshold
# there's 2 types of adaptive threshold (mean,gaussian)
# cv.AdaptiveThreshold(src, dst, maxValue, adaptive_method=CV_ADAPTIVE_THRESH_MEAN_C, thresholdType=CV_THRESH_BINARY, blockSize=3, param1)
mean_adaptive = cv.adaptiveThreshold(grayscale,255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 115, 1)
cv.imshow('mean_adaptive',mean_adaptive)

gaussian_adaptive = cv.adaptiveThreshold(grayscale,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 115, 1)
cv.imshow('gaussian_adaptive',gaussian_adaptive)



cv.waitKey(0)
cv.destroyAllWindows()
