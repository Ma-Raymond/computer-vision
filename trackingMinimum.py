import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
time.sleep(2)

mpHands = mp.solutions.hands  # Metapipe
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks) # This detects if we have a hand on screen
    
    if results.multi_hand_landmarks:
        for handItem in results.multi_hand_landmarks: # For each HAND!!
            for id,lm in enumerate(handItem.landmark):
                # print(id,lm) # id is the part of the hand, lm is the landmark of the hand
                h, w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) # Changes to pixels instead of ratios
                print(id,cx,cy) # This prints id of point on hand and then x y cords
                
                # This is the bottom of the hand!
                if id == 0:
                    cv2.circle(img,(cx,cy), 25, (255,0,255), cv2.FILLED)
                    
            mpDraw.draw_landmarks(img, handItem,mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3) # Puts fps on screen
    cv2.imshow('Image',img)
    cv2.waitKey(1)