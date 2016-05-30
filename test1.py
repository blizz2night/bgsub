# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:10:09 2016

@author: 宇烈
"""
import cv2
def test1():
    n=0
    for i in range(101):
        n+=i
    return n

def test2():
    return sum(range(101))


def test3():
    return sum(x for x in range(101))
#def sum2(x,y):
    #return lambda x+y
def test4():
    b
    return reduce(lambda x,y:x+y,range(101))
    
    
if __name__=='__main__':
    from timeit import Timer
    t1=Timer("test1()","from __main__ import test1")
    t2=Timer("test2()","from __main__ import test2")
    t3=Timer("test3()","from __main__ import test3")
    t4=Timer("test4()","from __main__ import test4")
#    print t1.timeit(1000000)
#    print t2.timeit(1000000)
    print t4.timeit(1000000)
#    print t3.timeit(1000000)
#    print t1.repeat(3,1000000)
#    print t2.repeat(3,1000000)
#    print t3.repeat(3,1000000)
