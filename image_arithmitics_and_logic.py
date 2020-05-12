import cv2
import numpy as np

image1 = cv2.imread('3D-Matplotlib.png')
image2 = cv2.imread('mainsvmimage.png')

addition = image1 + image2

#cv2.imshow('addition', addition)

# let's try the addition method which is built in cv
add = cv2.add(image1,image2)
#cv2.imshow('built_in_add', add)

# i can try imposing the images in another way
# waghited images
# cv2.addWeighted( frist_image ,its weight , second_image, its weight, gama value)
# don't fuck with the gama value and leave it alone as 0
weighted = cv2.addWeighted(image1,0.6, image2 , 0.4,0)
#cv2.imshow('weighted', weighted)

# now we are going to use ROI the region of image
# here i want to put the python logo on the lift of the add image
image3 = cv2.imread('mainlogo.png')

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = image3.shape
ROI = addition [0:rows, 0:cols]

# now we are going to create a mask which is a initial of conversion to the grayscale
image_gray = cv2.cvtColor(image3,cv2.COLOR_BGR2GRAY)
# now we are thresholding
# so any value above 220 will be converted into 255 which is white
# THRESH_BINARY_INV is going to do the inverse so what is lower than 220 will be converted
# into 0 which is black
# add a threshold
ret , mask = cv2.threshold(image_gray, 220 ,255, cv2.THRESH_BINARY_INV)
mask_inv = cv2.bitwise_not(mask)
cv2.imshow('mask_inv', mask_inv)

# Now black-out the area of logo in ROI
img1_background = cv2.bitwise_and(ROI,ROI,mask = mask_inv)
cv2.imshow('img1_background',img1_background)
# Take only region of logo from logo image.
im2_forground = cv2.bitwise_and(image3,image3,mask = mask)
cv2.imshow('im2_forground',im2_forground)
distination = cv2.add(img1_background,im2_forground)
cv2.imshow('distination', distination)
addition[0:rows,0:cols] = distination
cv2.imshow('result', addition)
#cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows(0)
