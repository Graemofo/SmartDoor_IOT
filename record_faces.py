#                  Step 2: Take photos to add to the dataset
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import time
import sqlite3
import os

conn = sqlite3.connect('database.db')
if not os.path.exists('./dataset'):
    os.makedirs('./dataset')
c = conn.cursor()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
camera.vflip = True
rawCapture = PiRGBArray(camera, size=(640, 480))

uname = raw_input("Enter your name: ")
c.execute('INSERT INTO users (name) VALUES (?)', (uname, ))
uid = c.lastrowid
sampleNum = 0
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image = frame.array

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.imwrite("dataset/User." + str(uid) + "." + str(sampleNum) + ".jpg", gray[y: y + h, x: x + w])
        print("Sample Pic: ", sampleNum)
        sampleNum = sampleNum + 1
        cv2.waitKey(100)
        cv2.imshow('record_faces', image)
        cv2.waitKey(3);
        time.sleep(0.5)
    if sampleNum > 100:
        break

    #cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
 
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
	    break

#cap.release()
conn.commit()
conn.close()
cv2.destroyAllWindows()
