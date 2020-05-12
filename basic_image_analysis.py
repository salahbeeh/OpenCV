import cv2
import numpy as np

image = cv2.imread('programming.jpg',cv2.IMREAD_COLOR)

image[55,55] = [255,255,255]
pixel = image[55,55]

# region of image  (ROI)
image[200:300,200:300]= [255,255,255]

pigwen_face = image[150:300,150:300]
image[0:150,0:150]= pigwen_face

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows(0)
