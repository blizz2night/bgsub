# -*- coding: utf-8 -*-
"""
Created on 2016/5/11

@author: 宇烈
"""
import cv2
import numpy as np
import time


def scaleResize(frame):
	size = frame.shape[0] * frame.shape[1]
	if size >= 614400:
		frame = cv2.resize(frame, (frame.shape[1] / 4, frame.shape[0] / 4))
		scale = 4
	elif size >= 307200:
		frame = cv2.resize(frame, (frame.shape[1] / 2, frame.shape[0] / 2))
		scale = 2
	else:
		scale = 1
	return frame, scale


def findContours(binaryImg, iteration=2, kernel=np.ones((3, 3), np.uint8), pixels=25):
	binaryImg = cv2.morphologyEx(binaryImg, cv2.MORPH_OPEN, kernel, iteration)
	binaryImg = cv2.dilate(binaryImg, kernel, iteration)
	# im2, contourList, hierarchy = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contourList, hierarchy = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contourList = filter(lambda x: cv2.contourArea(x) > pixels, contourList)
	return contourList


def findMaxContour(contourList):
	return reduce(lambda x, y: x if cv2.contourArea(x) > cv2.contourArea(y) else y, contourList)


def findRectangles(frame, contourList, ratio=0.002):
	rects = []
	if type(contourList) == list:
		if len(contourList) > 0:
			for contour in contourList:
				rect = cv2.boundingRect(contour)
				mul = frame.size / (rect[2] * rect[3])
				if 1 < mul and 1 / ratio > mul:
					rects.append(rect)
		else:
			raise ValueError, 'contours is null'
	else:
		raise ValueError, 'contours is not a list'
	return rects


def drawRectangle(frame, rect, scalar=(0, 0, 255)):
	frame = cv2.rectangle(frame, tuple(rect[0:2]), (rect[0] + rect[2], rect[1] + rect[3]), scalar, 3)
	return frame


def drawRectangles(frame, rects, scalar=(0, 0, 255)):
	if type(rects) == list:
		if len(rects) > 0:
			rectArray = np.array([])
			rectArray = np.array(rects)
			pts1 = rectArray[:, 0:2]
			pts2 = rectArray[:, 0:2] + rectArray[:, 2:4]
			for i in range(0, len(rectArray)):
				cv2.rectangle(frame, tuple(pts1[i]), tuple(pts2[i]), scalar, 3)
		else:
			raise ValueError, 'rects is null'
	else:
		raise ValueError, 'rects is not a list'
	return frame


def findCentres(rects):
	centrePoints = []
	if type(rects) == list:
		if len(rects) > 0:
			rectArray = np.array([])
			rectArray = np.array(rects)
			tempArray = rectArray[:, 0:2] + rectArray[:, 2:4] / 2
			centrePoints = tempArray.tolist()
		else:
			raise ValueError, 'rects is null'
	else:
		raise ValueError, 'rects is not a list'
	return centrePoints


# def drawPoints(frame, pointArray,scale=1,radius=2,scalar=(0,0,255)):
#    pointArray = pointArray*scale
#    for p in pointArray:
#        cv2.circle(frame,tuple(p),radius,scalar,-1)
#    return frame

class InvasionDetector(object):
	def __init__(self, ratio=0.002, learningRate=-1):
		# self.bgSubtractor = cv2.createBackgroundSubtractorKNN()
		self.bgSubtractor = cv2.BackgroundSubtractorMOG2()
		self.ratio = ratio
		self.learningRate = learningRate
		self.scale = 1
		self.fgmask = np.array([])
		# self.bgm = np.array([])
		self.contourList = []
		self.maxContour = []
		self.bbox = []
		self.contourRects = []
		self.ret = False

	def operate(self, frame, ratio=0.002, learningRate=-1):
		self.ratio = ratio
		self.learningRate = learningRate
		frame, self.scale = scaleResize(frame)
		# start = time.time()
		self.fgmask = self.bgSubtractor.apply(frame, self.fgmask, self.learningRate)
		# end = time.time()
		# print end-start
		# 连通区域
		self.contourList = findContours(self.fgmask)
		if len(self.contourList) > 0:
			self.contourRects = findRectangles(self.fgmask, self.contourList, self.ratio)
			if len(self.contourRects) > 0:
				self.ret = True
				tempArray = np.array(self.contourRects) * self.scale
				self.contourRects = tempArray.tolist()
			else:
				self.ret = False
		else:
			self.contourRects = []
			self.ret = False
			# 最大连通区域
			# self.maxContour = findMaxContour(self.contourList)
		#        rect = cv2.boundingRect(maxContour)
		#        rectArray = np.array(rect)*scale

		# maxRectArray = findRectangles(self.fgmask,self.maxContour,ratio)
		#        if len(maxRectArray)>0:
		#            self.ret = True
		#            bboxArray = maxRectArray[0]*self.scale
		#            self.bbox = bboxArray.tolist()
		#        else:
		#            self.ret = False
		#            self.bbox = []
		return self.ret, self.contourRects

	def getBgm(self):
		return self.bgSubtractor.getBackgroundImage()


class HoveringDetector(InvasionDetector):
	def __init__(self):
		self.tcount = 0
		self.fcount = 0
		#建模帧数
		self.count = 0
		self.ishovering = False
		super(HoveringDetector, self).__init__(0.002, 0.001)
	#interval徘徊时间
	def isHovering(self, frame, interval=200,modeling_frame_num=120, ratio=0.002, learningRate=0.0001):
		if self.count<=modeling_frame_num:
			self.count+=1
			ret, rects = super(HoveringDetector, self).operate(frame, ratio, learningRate=-1)
			return ret,rects
		else:
			ret, rects = super(HoveringDetector, self).operate(frame, ratio, learningRate)
			if ret:
				self.fcount = 0
				self.tcount += 1
				if self.tcount >= interval:
					self.ishovering = True
				else:
					self.ishovering = False
			else:
				if self.tcount > 0:
					if self.tcount >= interval:
						self.ishovering = True
					else:
						self.ishovering = False
					self.fcount += 1
					if self.fcount >= interval:
						self.tcount = 0
						self.fcount = 0
				else:
					self.ishovering = False
		return self.ishovering, rects


#cap = cv2.VideoCapture('1.avi')
# sensitivity = 300
cap = cv2.VideoCapture('rtsp://192.168.1.133:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif')
# cv2.namedWindow('frame')
# cv2.namedWindow('fgmask')
# bgSubtractor = cv2.createBackgroundSubtractorKNN()
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# learningRate = -1
# bgm_update_flag = False
# id = InvasionDetector()
hd = HoveringDetector()
# fgbg = cv2.createBackgroundSubtractorGMG()
# bgs = cv2.createBackgroundSubtractorMOG2()

firstFrame = True
canvas = np.array([], np.uint8)
while 1:
	ret, frame = cap.read()
	if not ret:
		break

	ret, rects = hd.isHovering(frame,modeling_frame_num=200,ratio=0.01)
	# ret,rects = id.operate(frame,0.002)
	if len(rects) > 0:
		drawRectangles(frame, rects)
		cps = findCentres(rects)
		# rects = findRectangles(id.fgmask,id.contourList)
		# drawRectangles(frame,rects)
		# contours = np.array(id.contourList)*id.scale
		# cv2.drawContours(frame,contours,-1,(0,255,0),3)
	#        if len(rects)>0:
	#            drawRectangles(frame,rects)

	# bgimg = id.getBgm()
	cv2.namedWindow('fg', 0)
	# cv2.namedWindow('bg', 0)
	cv2.imshow('fg', hd.fgmask)
	cv2.imshow('frame', frame)
	# cv2.imshow('bg', hd.getBgm())
	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break
	#        temp,scale = scaleResize(frame)
	#        #背景差分
	#        fgmask = bgSubtractor.apply(temp, learningRate)
	#        #连通区域
	#        contourList = findContours(fgmask)
	#        rectArray = findRectangles(fgmask,contourList, sensitivity)
	#        frame = drawRectangles(frame, rectArray, scale, (0,255,0))
	#        #最大连通区域
	#        maxContour = findMaxContour(contourList)
	#        maxRectArray = findRectangles(fgmask,maxContour,sensitivity)
	#        cpArray = findCentres(maxRectArray)
	#        frame = drawRectangles(frame, maxRectArray,scale)




	# drawPoints(canvas, cpArray, scale)
	# nzp = np.nonzero(canvas)[0:2]
	#        if(len(cpArray)>0):
	#            cpArray=cpArray*scale
	#            canvas[cpArray[:,1],cpArray[:,0]]=255
	#        cv2.imshow('canvas',canvas)
	#        nzp = np.nonzero(canvas)[0:2]
	#        frame[nzp]=(0,0,255)
	#        pt1 = rectArray[0:2]
	#        pt2 = rectArray[0:2]+rectArray[2:4]
	#        cv2.rectangle(frame, tuple(pt1),tuple(pt2),(0,0,255),3)
	# 连通区域外接矩形

	# bgmask = bgSubtractor.getBackgroundImage()
	# 尺度还原
	# rectArray = rectArray * scale

	# frame = drawRectangles(frame,rectArray)
	# centreArray = findCentres(rectArray)
	# for i in range(0,len(centreArray)):
	#    cv2.circle(canvas,tuple(centreArray[i]),2,(0,0,255),-1)
	# rp = np.nonzero(canvas)[0:2]
	# frame[rp]=(0,0,255)
	# cv2.imshow('bgmask', bgmask)


	# cv2.imshow('canvas',canvas)

# cv2.imshow('fgmask', fgmask)
cap.release()
cv2.destroyAllWindows()
