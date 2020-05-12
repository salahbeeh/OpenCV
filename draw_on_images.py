import cv2
import numpy as np

# just reading the image
image = cv2.imread('programming.jpg',cv2.IMREAD_COLOR)

# drawing a line
# cv2.line(image,srart point,end point,line_color, line width)
cv2.line(image,(5,5),(60,60),(255,255,255),5)

# drawing a rectangle
# cv2.rectangle(image,highest lift point ,lowest right point,rectangle_color, rectangle width)
cv2.rectangle(image,(100,100) ,(350,350),(0,255,0), 5)



cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
