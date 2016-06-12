# -*- coding: utf-8 -*-
"""
Created on 2016/5/11

@author: 宇烈
"""
import cv2
import numpy as np


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
	if type(contourList) == list:
		if len(contourList) > 1:
			return reduce(lambda x, y: x if cv2.contourArea(tuple(x)) > cv2.contourArea(tuple(y)) else y, contourList)
		elif len(contourList) == 1:
			return contourList[0]
		else:
			raise ValueError, 'contours is null'
	else:
		raise ValueError, 'contours is not a list'

def findMaxRect(rectList):
	if type(rectList) == list:
		if len(rectList) > 1:
			return reduce(lambda x, y: x if x[2]*x[3] > y[2]*y[3] else y, rectList)
		elif len(rectList) == 1:
			return rectList[0]
		else:
			raise ValueError, 'contours is null'
	else:
		raise ValueError, 'contours is not a list'

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
			centrePoints = map(lambda x:(x[0]+0.5*x[2],x[1]+0.5*x[3]),rects)
			# rectArray = np.array([])
			# rectArray = np.array(rects)
			# tempArray = rectArray[:, 0:2] + rectArray[:, 2:4] / 2
			# centrePoints = tempArray.tolist()
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
# 点
class Point(object):
	def __init__(self, x, y):
		self.x, self.y = x, y

# 向量
class Vector(object):
	def __init__(self, start_point=Point, end_point=Point):
		self.start, self.end = start_point, end_point
		self.x = end_point.x - start_point.x
		self.y = end_point.y - start_point.y

ZERO = 1e-9

def negative(vector):
    """取反"""
    return Vector(vector.end, vector.start)

def vector_product(vectorA, vectorB):
    '''计算 x_1 * y_2 - x_2 * y_1'''
    return vectorA.x * vectorB.y - vectorB.x * vectorA.y

def isIntersected(A, B, C, D):
    '''A, B, C, D 为 Point 类型'''
    AC = Vector(A, C)
    AD = Vector(A, D)
    BC = Vector(B, C)
    BD = Vector(B, D)
    CA = negative(AC)
    CB = negative(BC)
    DA = negative(AD)
    DB = negative(BD)

    return (vector_product(AC, AD) * vector_product(BC, BD) <= ZERO) \
        and (vector_product(CA, CB) * vector_product(DA, DB) <= ZERO)

class InvasionDetector(object):
	def __init__(self):
		# self.bgSubtractor = cv2.createBackgroundSubtractorKNN()
		self.bgSubtractor = cv2.BackgroundSubtractorMOG()
		self.count = 0
		self.path = []

	#入侵检测: ratio敏感度:目标占画面比
	def operate(self, frame, ratio=0.002, learningRate=-1):
		#self.ratio = ratio
		#self.learningRate = learningRate
		result = False
		frame, scale = scaleResize(frame)
		# start = time.time()
		fgmask = np.array([],np.uint8)
		fgmask = self.bgSubtractor.apply(frame, fgmask, learningRate)
		# end = time.time()
		# print end-start
		# 连通区域
		contourList = findContours(fgmask)
		if len(contourList) > 0:
			contourRects = findRectangles(fgmask, contourList, ratio)
			if len(contourRects) > 0:
				result = True
				tempArray = np.array(contourRects) * scale
				contourRects = tempArray.tolist()
		else:
			contourRects = []
		return result, contourRects, fgmask

	#单向入侵检测, pt1,pt2:绊线的位置, pt3,pt4检测入侵的方向, ratio敏感度:目标占画面比, interval:检测持续帧数
	def isInvaded(self, frame, pt1=(0.5, 0), pt2=(0.5, 1), pt3=(0, 0), pt4=(1, 0), ratio=0.002, learningRate=-1, interval=23):
		result = False
		rect = []
		rects =[]
		ret, rects = self.operate(frame, ratio, learningRate)
		if not ret:
			pass
		else:
			rect = findMaxRect(rects)
			pos = findCentres([rect])[0]
			self.path.append(pos)
			if self.count == 0:
				pass
			else:
				self.count += 1
				h, w = frame.shape[0:2]
				if type(pt1[0]) == float or type(pt1[1]) == float or type(pt2[0]) == float or type(pt2[1]) == float:
					ptC = Point(pt1[0]*w,pt1[1]*h)
					ptD = Point(pt2[0]*w,pt2[1]*h)
				else:
					ptC = Point(pt1[0],pt1[1])
					ptD = Point(pt2[0],pt2[1])
				ptA = Point(self.path[0][0], self.path[0][1])
				ptB = Point(pos[0], pos[1])
				ret = isIntersected(ptA, ptB, ptC, ptD)
				if not ret:
					pass
				else:
					dctvct = np.subtract(pt4, pt3)
					#print dctvct
					#spdvct = np.subtract(pos,self.lastpos)/interval
					spdvct = np.subtract(pos, self.path[0])
					#print spdvct
					#检测到单项入侵目标
					if np.dot(dctvct, spdvct) > 0:
						result = True
						#print 'invade!!',time.time()
					else:
						pass
		self.count += 1
		self.count %= interval
		if self.count == 0:
			self.path = []
		return result, rect, self.path

	def getBgm(self):
		return self.bgSubtractor.getBackgroundImage()


class HoveringDetector(InvasionDetector):
	def __init__(self):
		self.tcount = 0
		self.fcount = 0
		#建模帧数
		self.count = 0
		super(HoveringDetector, self).__init__(0.002, 0.001)
	#徘徊检测: interval徘徊持续帧数, warning报警持续帧数, modeling建模帧数, ratio敏感度:目标占画面比
	def isHovering(self, frame, interval=200, warning = 69, modeling_frame_num=120, ratio=0.002, learningRate=0.0001):
		result = False
		if self.count <= modeling_frame_num:
			self.count += 1
			ret, rects = super(HoveringDetector, self).operate(frame, ratio, learningRate=-1)
			return result, rects
		else:
			ret, rects = super(HoveringDetector, self).operate(frame, ratio, learningRate)
			if	np.count_nonzero(self.fgmask)/self.fgmask.size > 0.8:
				self.count = 0
				return result, rects
			if ret:
				self.fcount = 0
				self.tcount += 1
			else:
				if self.tcount > 0:
					self.fcount += 1
					if self.fcount >= warning:
						self.tcount = 0
						self.fcount = 0
				else:
					pass
			if self.tcount >= interval:
				result = True
			else:
				pass
		return result, rects


cap = cv2.VideoCapture('1.avi')

#cap = cv2.VideoCapture('rtsp://192.168.1.133:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif')
#入侵
id = InvasionDetector()
while 1:
	ret, frame = cap.read()
	if not ret:
		break
	ret, rects, fgmask = id.operate(frame)
	if len(rects) > 0:
		drawRectangles(frame, rects)
	if ret:
		print "invader"
	cv2.imshow('fg', fgmask)
	cv2.imshow('frame', frame)
	k = cv2.waitKey(10) & 0xff
	if k == 27:
		break
######

##单向越界
#id = InvasionDetector()
#while 1:
#	ret, frame = cap.read()
#	if not ret:
#		break
#	ret, rect, rects = id.isInvaded(frame,pt4=(-1,0),ratio=0.02)
#	if len(rect) > 0:
#		drawRectangles(frame,rects)
#		drawRectangle(frame, rect, (0, 255, 0))
#	if len(id.path)>1:
#		cv2.polylines(frame,np.array([id.path],np.int32),0,(255,0,0),thickness=2)
#	if ret:
#		print "invader"
#	cv2.imshow('fg', id.fgmask)
#	cv2.imshow('frame', frame)
#	k = cv2.waitKey(10) & 0xff
#	if k == 27:
#		break
#######

##徘徊
# hd = HoveringDetector()
# while 1:
# 	ret, frame = cap.read()
# 	if not ret:
# 		break
# 	ret, rects = hd.isHovering(frame,interval=50,modeling_frame_num=23,ratio=0.02)
# 	if len(rects) > 0:
# 		drawRectangles(frame, rects)
# 	if ret:
# 		print "hovering"
# 	cv2.imshow('fg', hd.fgmask)
# 	cv2.imshow('frame', frame)
# 	# cv2.imshow('bg', hd.getBgm())
# 	k = cv2.waitKey(10) & 0xff
# 	if k == 27:
# 		break
#######



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
