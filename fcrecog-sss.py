#!/usr/bin/python3

#importing the necessary packages (libraries)-importering de nødvendie pakkene

import subprocess
import numpy as np
import cv2
import os
import time
import sys
import math
import glob
import signal
from abdulFace import AbdulFace
import gtts
from gtts import gTTS 
from pydub import AudioSegment
from pydub.playback import play





check = 1


#importering bilder fra images

sfr = AbdulFace()
sfr.load_encoding_images("images/")


# Function 1- Starting camera capturing- kamera oppstart 

def Camera_start(wx,hx):
    global p
    rpistr = "libcamera-vid -t 0 --segment 1 --codec mjpeg -n -o /run/shm/test%06d.jpg --width " + str(wx) + " --height " + str(hx)
    p = subprocess.Popen(rpistr, shell=True, preexec_fn=os.setsid)

#  definering av variablene (initialise variables)
width        = 720
height       = 540
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_detected = 0
start = 0
cv2.namedWindow('Frame')
Text = "Left Mouse click on picture to EXIT, Right Mouse click for eye detaction ON/OFF"
ttrat = time.time()

Camera_start(width,height)




while True:
    
    if time.time() - ttrat > 3 and ttrat > 0:
        Text =""
        ttrat = 0
        
    # importering bildene 
    pics = glob.glob('/run/shm/test*.jpg')
    while len(pics) < 2:
        pics = glob.glob('/run/shm/test*.jpg')
    pics.sort(reverse=True)
    img = cv2.imread(pics[1])
    if len(pics) > 2:
        for tt in range(2,len(pics)):
            os.remove(pics[tt])
            
    # Putting rectangles around detected faces-Markering ansikter (Face)
    if check == 1: 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        face_detected = 0

        face_locations, face_names = sfr.detect_known_faces(img)
        for face_loc, name in zip(face_locations, face_names):
             y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
             
             # sett navne på bildene
             cv2.putText(img, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200), 4)
             
             # Create an audio file based on the customer's faces-skape en lydfil basert på kundens ansiktene
             tts =gtts.gTTS("velkommen kære kunde"+name, lang='no')
             tts.save("sounds/hello.mp3")
        
            
        
    # Showing images with rectangles and faces and their names-Fremvisning av bildene med å sette rektangel-rundt
    cv2.putText(img,Text, (10, height - 10), 0, 0.4, (0, 255, 255))
    
    # lydfil lesing (Spilling av lydfila-playing)
    song = AudioSegment.from_mp3("sounds/hello.mp3")
    play(song)
    
    cv2.waitKey(10)
