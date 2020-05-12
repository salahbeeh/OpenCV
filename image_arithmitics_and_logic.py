import cv2
import numpy as np

image1 = cv2.imread('3D-Matplotlib.png')
image2 = cv2.imread('mainsvmimage.png')

addition = image1 + image2

cv2.imshow('addition', addition)
cv2.waitKey(0)
cv2.destroyAllWindows(0)
