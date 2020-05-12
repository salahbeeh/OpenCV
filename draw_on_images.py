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

# drawing a circle
# cv2.circle(image , center, ridus,circle_color, width{if it -1 it will fill the circle})
cv2.circle(image, (200,200), 30, (255,0,0),3)


# polygones consist of points and you gonna connect those dots
# you have the option to colse the polygone
# the points for the polygone
points = np.array([[50,45],[60,47],[69,78],[39,85],[82,64]],np.int32)
points = points.reshape((-1,1,2))
#cv2.polylines(image, [points],wither or not to connect the first point to  the last one, color, line width)
cv2.polylines(image,[points],True,(0,255,255),6)

# writing on the image
# define a font
font = cv2.FONT_HERSHEY_SIMPLEX
#                            start,font, size,color, space between charcters, alusing 
cv2.putText(image,'oh! shit',(0,130),font, 1, (100,220,150), 2,cv2.LINE_AA)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
