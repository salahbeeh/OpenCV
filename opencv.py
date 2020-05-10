import cv2
import numpy as np
import matplotlib as plt

# careful with the identions

"""
# reading an image can be done in opencv with many ways:
# IMREAD_GRAYSCALE , we can basicly just replace it with 0
# IMREAD_COLOR , or just replace it with 1
 """

# reading a randam image from my labtop using opencv
image = cv2.imread('programming.jpg',cv2.IMREAD_GRAYSCALE)

# showing the image i just read
cv2.imshow('Image' , image)

# litarly waiting for any button to be presed
cv2.waitkey(0)

cv2.destroyAllWindows()
