import cv2
import mediapipe as mp
import time


class handDetection():
    def __init__(self,mode=False,maxHands =2, detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.dectectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands  # Metapipe
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,1,self.dectectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    
    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks) # This detects if we have a hand on screen
        
        if self.results.multi_hand_landmarks:
            for handItem in self.results.multi_hand_landmarks: # For each HAND!!
                if draw:  
                    self.mpDraw.draw_landmarks(img,handItem,self.mpHands.HAND_CONNECTIONS)
                
                
        return img
    def findPosition(self,img,handNo=0,draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                # print(id,lm) # id is the part of the hand, lm is the landmark of the hand
                h, w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) # Changes to pixels instead of ratios
                # print(id,cx,cy) # This prints id of point on hand and then x y cords
                lmList.append([id,cx,cy])
                # This is the bottom of the hand!
                if draw:
                    if (id == 4):
                        cv2.circle(img,(cx,cy), 2, (255,0,255), cv2.FILLED)
        return lmList
        
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetection()
    while True:
        success, img = cap.read()
        
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        
        # FPS Finder
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3) # Puts fps on screen
        cv2.imshow('Image',img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()