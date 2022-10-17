import cv2 
import numpy as np 
 
webcam = cv2.VideoCapture(0) 
 
def colordetect(frame): 
    hsvFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) 
 
    # ORANGE
    orangeLower = np.array([0,87,111], np.uint8) 
    orangeUpper = np.array([22,255,255], np.uint8) 
    orangeMask = cv2.inRange(hsvFrame, orangeLower, orangeUpper) 
 
    kernel = np.ones((5,5), "uint8") 
 
    orangeMask = cv2.dilate(orangeMask, kernel) 
    cont,hier = cv2.findContours(orangeMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
 
    for pic,cont in enumerate(cont): 
        area = cv2.contourArea(cont) 
        if(area > 1000): 
            x,y,w,h = cv2.boundingRect(cont) 
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,127,255), 2)  
 
            cv2.putText(frame, "Orange Color", (x,y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,127,255)) 
 
    # YELLOW
    yellowLower = np.array([22,87,111], np.uint8) 
    yellowUpper = np.array([38,255,255], np.uint8) 
    yellowMask = cv2.inRange(hsvFrame, yellowLower, yellowUpper) 
 
    kernel = np.ones((5,5), "uint8") 
 
    yellowMask = cv2.dilate(yellowMask, kernel) 
    cont,hier = cv2.findContours(yellowMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
 
    for pic,cont in enumerate(cont): 
        area = cv2.contourArea(cont) 
        if(area > 1000): 
            x,y,w,h = cv2.boundingRect(cont) 
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)  
 
            cv2.putText(frame, "Yellow Color", (x,y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,255,255)) 
 
    # GREEN
    greenLower = np.array([38,87,111], np.uint8) 
    greenUpper = np.array([75,255,255], np.uint8) 
    greenMask = cv2.inRange(hsvFrame, greenLower, greenUpper) 
 
    kernel = np.ones((5,5), "uint8") 
 
    greenMask = cv2.dilate(greenMask, kernel) 
    cont,hier = cv2.findContours(greenMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
 
    for pic,cont in enumerate(cont): 
        area = cv2.contourArea(cont) 
        if(area > 1000): 
            x,y,w,h = cv2.boundingRect(cont) 
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)  
 
            cv2.putText(frame, "Green Color", (x,y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,255,0)) 
 
    # BLUE
    blueLower = np.array([75,87,111], np.uint8) 
    blueUpper = np.array([130,255,255], np.uint8) 
    blueMask = cv2.inRange(hsvFrame, blueLower, blueUpper) 
 
    kernel = np.ones((5,5), "uint8") 
 
    blueMask = cv2.dilate(blueMask, kernel) 
    cont,hier = cv2.findContours(blueMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
   
    for pic,cont in enumerate(cont): 
        area = cv2.contourArea(cont) 
        if(area > 1000): 
            x,y,w,h = cv2.boundingRect(cont) 
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)  
 
            cv2.putText(frame, "Blue Color", (x,y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0)) 
    # PURPLE
    purpLower = np.array([130,87,111], np.uint8) 
    purpUpper = np.array([160,255,255], np.uint8) 
    purpMask = cv2.inRange(hsvFrame, purpLower, purpUpper) 
 
    kernel = np.ones((5,5), "uint8") 
 
    purpMask = cv2.dilate(purpMask, kernel) 
    cont,hier = cv2.findContours(purpMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
     
    for pic,cont in enumerate(cont): 
        area = cv2.contourArea(cont) 
        if(area > 1000): 
            x,y,w,h = cv2.boundingRect(cont) 
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,127), 2)  
 
            cv2.putText(frame, "Purple Color", (x,y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,0,127)) 
 
    # RED
    redLower = np.array([160,87,111], np.uint8) 
    redUpper = np.array([179,255,255], np.uint8) 
    redMask = cv2.inRange(hsvFrame, redLower, redUpper) 
 
    kernel = np.ones((5,5), "uint8") 
 
    redMask = cv2.dilate(redMask, kernel) 
    cont,hier = cv2.findContours(redMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
 
 
    for pic,cont in enumerate(cont): 
        area = cv2.contourArea(cont) 
        if(area > 1000): 
            x,y,w,h = cv2.boundingRect(cont) 
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)  
 
            cv2.putText(frame, "Red Color", (x,y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255)) 
    return frame 
 
while True: 
    ret,frame = webcam.read(0) 
    frame = colordetect(frame) 
    cv2.imshow("Color Detection",frame) 
    if cv2.waitKey(10) & 0xFF == ord("e"): 
        webcam.read() 
        cv2.destroyAllWindows 
        break