import cv2
import numpy as np

image1 = cv2.imread('3D-Matplotlib.png')
image2 = cv2.imread('mainsvmimage.png')

addition = image1 + image2

cv2.imshow('addition', addition)

# let's try the addition method which is built in cv
add = cv2.add(image1,image2)
cv2.imshow('built_in_add', add)

# i can try imposing the images in another way
# waghited images
# cv2.addWeighted( frist_image ,its weight , second_image, its weight, gama value)
# don't fuck with the gama value and leave it alone as 0
weighted = cv2.addWeighted(image1,0.6, image2 , 0.4,0)
cv2.imshow('weighted', weighted)
cv2.waitKey(0)
cv2.destroyAllWindows(0)
