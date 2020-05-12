import cv2
import numpy as np

image1 = cv2.imread('3D-Matplotlib.png')
image2 = cv2.imread('mainsvmimage.png')

addition = image1 + image2

cv2.imshow('addition', addition)

# let's try the addition method which is built in cv
add = cv2.add(image1,image2)
cv2.imshow('built_in_add', add)
cv2.waitKey(0)
cv2.destroyAllWindows(0)
