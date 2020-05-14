import numpy as np
import cv2

img = cv2.imread('corners.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
# cv2.goodFeaturesToTrack(image ,num corners to be detected , corner quality, minDistance)
# minDistance = Minimum possible Euclidean distance between the returned corners.
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)
for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),3,255,-1)

cv2.imshow('corner',img)
if cv2.waitKey(0) & 0xff ==27:
    cv2.destroyAllWindows()
