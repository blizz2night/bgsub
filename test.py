# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:20:39 2016

@author: 宇烈
"""

import math
import numpy as np

a = (1,1)
b=(-1,1)
sin = np.cross(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
cos = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

#print math.degrees(math.atan2(60,60))
#print math.atan(1)
print np.rad2deg(np.arctan2(sin,cos))
print np.intersect1d([1,1,0,0],[-1,1,0,0])