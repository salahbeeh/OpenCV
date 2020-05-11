import cv2
import numpy as np

# 0 means that you are capturing from the first cam
# 1 means that you are capturing from the second cam
# if you have 2 or whatever
video = cv2.VideoCapture(1)

# this the the codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# the name and the size of the video i'm trying to capture
out = cv2.VideoWriter('output.avi', fourcc,20.0, (640,480))

while True:
    # reading the rgb video
    ret, frame = video.read()
    # reading the rgb video and convert it into grayscale
    gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
    # well, writing means 'saving' pretty much simple
    out.write(frame)

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    # the video will wait for any key to be pushed
    # but with '&' it will wait for the the hecadecimal of the key q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()
