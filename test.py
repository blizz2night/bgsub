# -*- coding: utf-8 -*-
"""
Created on 2016/5/11

@author: 宇烈
"""
import cv2
import numpy as np

def scaleResize(frame):
    size = frame.shape[0]*frame.shape[1]
    if size>=614400:
        frame = cv2.resize(frame, (frame.shape[1]/4,frame.shape[0]/4))
        scale=4
    elif size>=307200:
        frame = cv2.resize(frame, (frame.shape[1]/2,frame.shape[0]/2))
        scale=2
    else:
        scale=1
    return frame,scale
    
def findRectangles(bgSubtractor,frame, sensitivity = 300, kernel = np.ones((3,3),np.uint8)):
    fgmask = bgSubtractor.apply(frame)
    #bgmask = fgbg.getBackgroundImage()
    
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    fgmask = cv2.dilate(fgmask, kernel, iterations=3)
    im2, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects=[]
#    rectArray = np.array([],np.int32)
    if len(contours)>0:
        for contour in contours:
            rect = cv2.boundingRect(contour)
            ratio = fgmask.size/(rect[2]*rect[3])
            if ratio>10 and ratio<sensitivity:
                rects.append(rect)
#                if len(rects)>0:
#                    rectArray = np.array(rects)     
    rectArray = np.array(rects,np.int32)
    return rectArray
    
def drawRectangles(frame, rectArray):
    if len(rectArray)>0:
        pts1 = rectArray[:,0:2]
        pts2 = rectArray[:,0:2]+rectArray[:,2:4]
    for i in range(0,len(rectArray)):
        cv2.rectangle(frame, tuple(pts1[i]),tuple(pts2[i]),(0,0,255),3)
    return frame
    
def findCentre(rectArray):
    centreArray = np.array([],np.int32)
    if len(rectArray)>0:
        centreArray = rectArray[:,0:2]+rectArray[:,2:4]/2
    return centreArray

def drawPoints(frame, points,radius=2,scalar=(0,0,255)):
    for i in range(0,len(points)):
        cv2.circle(frame,tuple(points[i]),radius,scalar,-1)
    return frame
    
cap = cv2.VideoCapture('1.avi')
#cap = cv2.VideoCapture('rtsp://192.168.1.133:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif')
#cv2.namedWindow('frame')
#cv2.namedWindow('fgmask')
bgSubtractor = cv2.createBackgroundSubtractorKNN()
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

learningRate = 0.1
#bgm_update_flag = False

#fgbg = cv2.createBackgroundSubtractorGMG()
#bgs = cv2.createBackgroundSubtractorMOG2()
count=1
canvas = np.array([],np.uint8)
while 1:
    if count == 1:
        count = count + 1
        ret, frame = cap.read()
        if not ret:
            pass
        canvas = np.zeros(frame.shape,np.uint8)
    else:
        ret, frame = cap.read()
        if not ret:
            break
        temp,scale = scaleResize(frame)
    #    if frame.size>307200:
    #        frame = cv2.resize(frame, (640, 480))
        rectArray = findRectangles(bgSubtractor,temp)
        rectArray = rectArray * scale
        frame = drawRectangles(frame,rectArray)
        centreArray = findCentre(rectArray)
        for i in range(0,len(centreArray)):
            cv2.circle(canvas,tuple(centreArray[i]),2,(0,0,255),-1)
    
        rp = np.nonzero(canvas)[0:2]
        frame[rp]=(0,0,255)
        cv2.imshow('frame', frame)
        cv2.imshow('canvas',canvas)
    
#    cv2.imshow('fgmask', fgmask)
#    cv2.imshow('bgmask', bgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
