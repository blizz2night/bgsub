# -*- coding: utf-8 -*-
"""
Created on 2016/5/11

@author: 宇烈
"""
import cv2
import numpy as np
cap = cv2.VideoCapture('1.avi')
#cap = cv2.VideoCapture('rtsp://192.168.1.133:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif')
#cv2.namedWindow('frame')
#cv2.namedWindow('fgmask')
fgbg = cv2.createBackgroundSubtractorKNN()
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel = np.ones((3,3),np.uint8)
learningRate = 0.1
#bgm_update_flag = False

#fgbg = cv2.createBackgroundSubtractorGMG()
#bgs = cv2.createBackgroundSubtractorMOG2()

while 1:
    ret, frame = cap.read()
    if not ret:
        break;
    #frame.shape
    #frame = cv2.resize(frame, ())
    if frame.size>307200:
        frame = cv2.resize(frame, (640, 480))
    fgmask = fgbg.apply(frame)
    bgmask = fgbg.getBackgroundImage()
    
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    fgmask = cv2.dilate(fgmask, kernel, iterations=3)
    im2, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects=[]
    if len(contours)>0:
        for contour in contours:
            rect = cv2.boundingRect(contour)
            ratio = fgmask.size/(rect[2]*rect[3])
            if ratio>10 and ratio<300:
                rects.append(rect)
        #cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        if len(rects)>0:
            rectArray = np.array(rects)
            pts1 = rectArray[:,0:2]
            pts2 = rectArray[:,0:2]+rectArray[:,2:4]
            for i in range(0,len(rects)):
                cv2.rectangle(frame, tuple(pts1[i]),tuple(pts2[i]),(0,0,255),3)
            
    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    cv2.imshow('bgmask', bgmask)
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
