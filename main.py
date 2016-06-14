# -*- coding: utf-8 -*-
import cv2
import numpy as np
import enum
import math
import bgsub

PainterType = enum.Enum('PainterType', ('ROI', 'TRIPWIRE', 'VECTOR'))

class MousePainter:
	def __init__(self, sp=(-1,-1),ep=(-1,-1)):
		self.sp = sp
		self.ep = ep
		self.ondraw_flag = False
		self.ontrack_flag = False
	def clear(self):
		self.ondraw_flag = False
		self.ontrack_flag = False

def drawArrow(img, p1, p2, color=(0,255,0), thickness=2):
	angle = math.atan2(p1[1]-p2[1], p1[0]-p2[0])
	#print angle
	cv2.line(img, p1, p2, color, thickness)
	x = int(p2[0] + 20 * math.cos(angle + math.pi * 30 / 180))
	y = int(p2[1] + 20 * math.sin(angle + math.pi * 30 / 180))
	cv2.line(img, p2, (x,y), color, thickness)
	x = int(p2[0] + 20 * math.cos(angle - math.pi * 30 / 180))
	y = int(p2[1] + 20 * math.sin(angle - math.pi * 30 / 180))
	cv2.line(img, p2, (x,y), color, thickness)

class Painter:
	type = PainterType.ROI
	roi = MousePainter()
	tripwire = MousePainter()
	vector = MousePainter()
	@classmethod
	def draw(cls, img):
		if cls.roi.ondraw_flag:
			cv2.rectangle(img, cls.roi.sp, cls.roi.ep, (255, 0, 0), 2)
		if cls.tripwire.ondraw_flag:
			cv2.line(img, cls.tripwire.sp, cls.tripwire.ep, (0, 0, 255), 2)
		if cls.vector.ondraw_flag:
			#cv2.line(img, cls.vector.sp, cls.vector.ep, (0, 255, 0), 2)
			drawArrow(img, cls.vector.sp, cls.vector.ep, color=(0, 255, 0), thickness=2)

def trackMouse(event, x, y, flags, param):
	painter = param
	if event == cv2.EVENT_LBUTTONDOWN:
		painter.ondraw_flag = False
		painter.ontrack_flag = True
		painter.sp = (x, y)
	elif event == cv2.EVENT_MOUSEMOVE:
		if painter.ontrack_flag:
			painter.ep = (x, y)
			painter.ondraw_flag = True
	elif event == cv2.EVENT_LBUTTONUP:
		if painter.ontrack_flag:
			painter.ep = (x, y)
			painter.ontrack_flag = False
	elif event == cv2.EVENT_RBUTTONDOWN:
		painter.clear()

def handleMouseEvent(event, x, y, flags, param):
	Painter = param
	if Painter.type == PainterType.ROI:
		trackMouse(event, x, y, flags, Painter.roi)
		# if Painter.roi.ondraw_flag and ~Painter.roi.ontrack_flag:
		# 	Painter.type = PainterType.TRIPWIRE
	elif Painter.type == PainterType.TRIPWIRE:
		trackMouse(event, x, y, flags, Painter.tripwire)
		# if Painter.tripwire.ondraw_flag and ~Painter.tripwire.ontrack_flag:
		# 	Painter.type = PainterType.VECTOR
	elif Painter.type == PainterType.VECTOR:
		trackMouse(event, x, y, flags, Painter.vector)

def getRoiArray(img, p1, p2):
	if p1[0] < p2[0]:
		spx = p1[0]
		epx = p2[0]
	else:
		spx = p2[0]
		epx = p1[0]
	if p1[1] < p2[1]:
		spy = p1[1]
		epy = p2[1]
	else:
		spy = p2[1]
		epy = p1[1]
	return img[spy:epy, spx:epx]

def getRoiRect(img, p1, p2):
	if p1[0] < p2[0]:
		spx = p1[0]
		epx = p2[0]
	else:
		spx = p2[0]
		epx = p1[0]
	width = epx - spx
	if p1[1] < p2[1]:
		spy = p1[1]
		epy = p2[1]
	else:
		spy = p2[1]
		epy = p1[1]
	height = epy - spy
	return (spx,spy,width,height)

painter = MousePainter()
id = bgsub.InvasionDetector()
hd = bgsub.HoveringDetector()
dct_flag = False
DetectorType = enum.Enum('DetectorType',('INVASION','HOVERING'))
dct = DetectorType.HOVERING
cv2.namedWindow('image')
cv2.setMouseCallback('image', handleMouseEvent, Painter)
cap = cv2.VideoCapture('rtsp://192.168.1.133:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif')
rects = []
path = []
fgmask = []
wcount = 0
filecount = 0
while(1):
	ret, frame = cap.read()
	if dct == DetectorType.INVASION:
		if dct_flag:
			roi = getRoiRect(frame, Painter.roi.sp, Painter.roi.ep)
			#print roi
			#ret, rects, fgmask = id.operateROI(frame, roi)
			ret, rect, path, fgmask = id.isInvadedROI(frame, roi, Painter.tripwire.sp, Painter.tripwire.ep, Painter.vector.sp, Painter.vector.ep, interval = 69)

				#print rects
				# for rect in rects:
				# 	cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 2)
			if len(rect)>0:
				cv2.rectangle(frame, (rect[0],rect[1]),(rect[0]+rect[2], rect[1]+rect[3]), (0,0,255),2)
				#print path
			if len(path) > 1:
				cv2.polylines(frame, np.array([path], np.int32), 0, (255, 255, 0), 2)
			if ret:
				wcount = 1
			cv2.imshow('fg', fgmask)
		Painter.draw(frame)
		if wcount > 0:
			cv2.putText(frame, 'Invader!!!', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
			wcount += 1
			wcount %= 69
			cv2.imwrite(str(filecount) + '.jpg', frame)
			filecount += 1
	elif dct == DetectorType.HOVERING:
		if dct_flag:
			roi = getRoiRect(frame, Painter.roi.sp, Painter.roi.ep)
			# print roi
			# ret, rects, fgmask = id.operateROI(frame, roi)
			ret, rects = hd.isHoveringROI(frame, roi, ratio=0.01)
			# print rects
			for rect in rects:
				cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 2)
			if ret:
				wcount = 1
		Painter.draw(frame)
		if wcount > 0:
			cv2.putText(frame, 'Hovering!!!', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
			wcount += 1
			wcount %= 69
			cv2.imwrite(str(filecount) + '.jpg', frame)
			filecount += 1
  	cv2.imshow('image', frame)

	k = cv2.waitKey(5)
	if k & 0xFF == 27:
		break
	elif k & 0xFF == 13:
		dct_flag = True
	elif k & 0xFF == 99:
		if Painter.type == PainterType.ROI:
			Painter.type = PainterType.TRIPWIRE
		elif Painter.type == PainterType.TRIPWIRE:
			Painter.type = PainterType.VECTOR
		elif Painter.type == PainterType.VECTOR:
			Painter.type = PainterType.ROI
cv2.destroyAllWindows()