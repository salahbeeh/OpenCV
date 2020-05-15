import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    """
    virtual void cv::BackgroundSubtractorMOG2::apply 	(
        InputArray :image,
		OutputArray : 	fgmask,
		double  	learningRate = -1
	)

Python:
	fgmask	=	cv.BackgroundSubtractorMOG2.apply(	image[, fgmask[, learningRate]]	)

Computes a foreground mask.

Parameters
    image:	      Next video frame. Floating point frame will be used without scaling
                  and should be in range [0,255].
    fgmask:	      The output foreground mask as an 8-bit binary image.
    learningRate: The value between 0 and 1 that indicates how fast the background
                  model is learnt. Negative parameter value makes the algorithm to
                  use some automatically chosen learning rate. 0 means that the
                  background model is not updated at all, 1 means that the background
                  model is completely reinitialized from the last frame.

Implements cv::BackgroundSubtractor.
"""


    fgmask = fgbg.apply(frame)

    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',fgmask)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
