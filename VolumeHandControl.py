import cv2
import time
import numpy as np
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import trackingModule as tm # this has classes

####################
wCam,hCam = 640,488
####################
cap = cv2.VideoCapture(0) # Getting the capture from camera

cap.set(3,wCam)
cap.set(4,hCam)

pTime = 0
cTime = 0

detector = tm.handDetection(detectionCon=0.7)

############################ Sound dependances
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
############################

# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

while True:
    success, img = cap.read()
    img = detector.findHands(img) # Puts the image as the find hand img
    
    lmList = detector.findPosition(img,draw=False)
    if len(lmList) != 0:
        print(lmList[4],lmList[8])
        
        x1,y1 = lmList[4][1], lmList[4][2] # This sets the x and y of point 4
        x2,y2 = lmList[8][1], lmList[8][2] # This sets the x and y of point 4
        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED) # This created the two circles on the 4 points I want

        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),2)
        centerX,centerY = (x1+x2)//2,(y1+y2)//2 # This gets the middle coordinates
        
        cv2.circle(img,(centerX, centerY),15,(255,0,0),cv2.FILLED) # Center circle
        
        length = math.hypot(x2-x1,y2-y1)
        
        
        # Hand range 50 - 300
        # Vol range = -63 - 0
        
        vol = np.interp(length,[50,300],[minVol,maxVol]) # Sets the length based on the min to maxvol
        volume.SetMasterVolumeLevel(vol, None)
        
        if length <50:
            cv2.circle(img,(centerX, centerY),15,(0,255,0),cv2.FILLED) # Center circle
            
        
    ####### Getting the FPS ##########
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    ##################################
    
    # Showing the text on screen
    cv2.putText(img,f'FPS:{int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX,
                1,(255,0,255),3) # Scale , Color, Thickness
    
    cv2.imshow("Img",img)
    cv2.waitKey(1)
    