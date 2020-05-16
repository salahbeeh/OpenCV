import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)

while 1:
    # returns the feed from the camera
    ret, img = cap.read()
    # convert the feed into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # HaarDetectObjects(image, scale_factor=1.3, min_neighbors=5)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    for (x,y,w,h) in faces:
        # drawing a rectangle around the face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # writing on the first pixel of the rectangle
        cv2.putText(img,'Face',(x,y), font, 0.5, (11,255,255), 2, cv2.LINE_AA)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # HaarDetectObjects(image)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            # drawing a rectangle around the eyes
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            # writing on the first pixel of the rectangle
            cv2.putText(img,'Eye',(ex+x,ey+y), font, 0.5, (11,0,255), 2, cv2.LINE_AA)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
