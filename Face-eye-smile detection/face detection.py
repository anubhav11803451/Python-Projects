""" In this Project I will be using 'haarcascade_frontalface_default.xml,haarcascade_eye.xml,haarcascade_smile.xml'
cascade by INTEL to detect eyes , faces and smiles"""
"""Author : Anubhav Gupta"""

import cv2
import numpy as np

f_casacade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#CascadeClassifier is used to select the features from xml file
e_casacade=cv2.CascadeClassifier('haarcascade_eye.xml')
s_cascade= cv2.CascadeClassifier('haarcascade_smile.xml')


capture = cv2.VideoCapture(0) #VideoCapture is a constructor which will be call by release() function below
                            #i took 0 launch camera of my pc


while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame,1)#flip function is used here to flip to 2D array around vertical, horizontal, or both
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#cvtColor fun converts an image from one color to another(*channel order sholud be(RBG or BGR) in OpenCV format is RGB but in actual it is BGR.)

    faces = f_casacade.detectMultiScale(gray,1.3,5)#to detect the face in image and returned as a list(1.1)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+w),(255,255,0),2)#to draw the rectangle around the face starting point(x,y) and end point(x+w,y+w)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = e_casacade.detectMultiScale(roi_color,scaleFactor=1.3,minNeighbors=5,minSize=(5,5),)#(same as 1.1 but eyes rather than face)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew,ey+eh),(0,255,0),2)#to draw rectangle around your eyes

        smiles = s_cascade.detectMultiScale(roi_color,scaleFactor=1.3,minNeighbors=15,minSize=(25,25),)

        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)#to draw rectangle around your smile


        cv2.imshow('video',frame)#the function imshow is used to display an image in the specified window.

    key = cv2.waitKey(30) & 0xff#waitkey fun  is used to waits for a key event infinitely or for dely
    if key == 27: #press ESC key to quit
        break

capture.release()#release fun will call the VideoCapture constructor above
cv2.destroyAllWindows()
