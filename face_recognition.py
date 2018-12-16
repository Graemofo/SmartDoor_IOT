#                        Step 4: Run facial recognition
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import sqlite3
import os
import RPi.GPIO as GPIO

GPIO.setwarnings(False)

GPIO.setmode(GPIO.BCM) # Broadcom pin-numbering scheme

GPIO.setup(17, GPIO.OUT) # output GPIO is set to GPIO pin 17

GPIO.output(17, GPIO.HIGH)


conn = sqlite3.connect('database.db')
c = conn.cursor()

fname = "recognizer/trainingData.yml"
if not os.path.isfile(fname):
    print("Please train the data first")
    exit(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(fname)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

liveFeed = "http://localhost:8000/index"
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 18 #32
#camera.vflip = True
rawCapture = PiRGBArray(camera, size=(640, 480))
GPIO.output(17, GPIO.HIGH)
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image = frame.array
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
        ids,conf = recognizer.predict(gray[y:y+h,x:x+w])
        c.execute("select name from users where id = (?);", (ids,))
        result = c.fetchall()
        name = result[0][0]
        if conf < 70:  # If confidence is 30% or more
            cv2.putText(image, name, (x+2,y-20), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0))
            
            GPIO.output(17, GPIO.LOW)
            time.sleep(30)
            
            
        else:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)

        GPIO.output(17, GPIO.HIGH)
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
            GPIO.cleanup()
	    break
