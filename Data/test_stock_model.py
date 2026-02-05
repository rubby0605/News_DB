#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 10:10:13 2022

@author: rubylintu
"""
import math
import random
import matplotlib.pyplot as plt

class sc:
    def __init__(self, bp, net, time, amount):
        self.bp = bp
        self.net = net
        self.time = time
        self.amount = amount

class particle(sc):
    def get_2dnpdf(self,mu_x, mu_y, sgm_x, sgm_y):
        zz = 0
        while True:
            xx = (random.random()*5-2.5)*mu_x
            yy = (random.random()*5)*mu_y
            zz = random.random()
            fxy = 1/2/math.pi/sgm_x/sgm_y * math.exp(-1/2*(pow(((xx-mu_x)/sgm_x),2)+pow((yy-mu_y/sgm_y),2))) 
            if zz <= fxy:
                break
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self.fxy = fxy
    def plot(self):
        plt.plot(self.xx, self.yy,'k+')
        
        
MUX = 5
MUY = 1
SGM_X = 5
SGM_Y = 2
for i in range(1000):        
    aa= particle(620+random.random()*4-2,random.random()*5,random.random(), int(random.random()*3))
    aa.get_2dnpdf(MUX, MUY, SGM_X, SGM_Y)
    aa.plot()