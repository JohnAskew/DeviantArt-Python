import os, sys
try:
    import  cv2
except:
    os.system('pip install opencv-contrib-python')
    import cv2
try:
    import numpy as np
except:
    os.system('pip install numpy')
    import numpy as np


people_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1) # You may need to set the "1" to 0 if you are using the installed webcam. Don't be afraid to experiment!

while True:
    ret, img = cap.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    people = people_cascade.detectMultiScale(gray)
    for (x, y, w, h) in people:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()