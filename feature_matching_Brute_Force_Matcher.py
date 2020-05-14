import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('BMW1.jpg',0)
img2 = cv2.imread('logos.jpg',0)

# Initiate SIFT detector
# SIFT = Scale-invariant feature transform
# SIFT detectors allow us to know the matches evev if the 2 images we are
# comparing are not the same or the matched objects are rotated it still find
# the matches unless the object feature are too small to be matched
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# BFMatche returns the best match.
# NORM_HAMMING Hamming use distance as measurement
"""
crossCheck which is false by default. If it is true, Matcher returns only those
matches with value (i,j) such that i-th descriptor in set A has j-th descriptor
in set B as the best match and vice-versa
"""
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

# drawing the matches
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
plt.imshow(img3)
plt.show()
