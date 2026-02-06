#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""  plot price
Created on Mon Nov 15 10:18:17 2021

@author: rubylintu
"""

import matplotlib.pyplot as plt
import re
import os, sys
from datetime import date
import numpy as np

num = sys.argv[1]  #python3 plot_price.py 2330
datestr = str(date.today().year)+str(date.today().month)+str(date.today().day)
f=open("Data/"+num+"_"+datestr[2:]+".txt")
lines = f.readlines()
mat_dealamount = np.zeros(len(lines))
mat_price = np.zeros(len(lines))
mat_timetag = np.chararray([len(lines), 2],itemsize=100)
mat_timetag[:,:]='...'
i = 0
kk = 0
price=0
for line in lines:
    if(':' not in line):
        line0 = re.sub("'","",line)
        line1 = re.sub("-","0",line0)
        line2 = re.split('\[|\]|,|\n',line1)
        num = line2[1]
        name = line2[2]
        deal = line2[3]
        dealamount = line2[4]
        wholedealamount = line2[5]
        startprice = line2[6]
        highestprice = line2[7]
        lowestprice = line2[8]
        yestprice = line2[9]
    else:
        line0 = re.sub("'","",line)
        line2 = re.split(':| |-|,|\n',line0)
        yy = int(line2[2])
        mm = int(line2[3])
        dd = int(line2[4])
        hh = int(line2[5])
        mt = int(line2[6])
        ss = float(line2[7])
        line3 = re.split(' |\n',line)
        if (hh >= 9) and (hh<13 or (hh==13 and mt <=30)):
            mat_price[kk] = float(deal)
            mat_dealamount[kk] = float(dealamount)
            mat_timetag[kk,0] = str(mm)+'/'+str(dd)
            mat_timetag[kk,1] = str(hh)+':'+str(mt)
            kk = kk + 1
    i = i + 1
plt.figure(figsize=((10,8)))
plt.subplot(1,2,1)
plt.plot(range(kk-2),mat_price[1:kk-1],'k.')
plt.axis([0, kk-2, max(mat_price[:])-30, max(mat_price[:])+5])
for i in range(0,kk-2,int((kk-2)/5)):
    plt.text(i, mat_price[i]*0.95, mat_timetag[i,1].decode('big5'))
    plt.text(i, mat_price[i]*0.95-5, mat_timetag[i,0].decode('big5'))
plt.legend(['price'])
plt.subplot(1,2,2)
plt.axis([0, kk-2, min(mat_dealamount[:])-5, max(mat_dealamount[:])+5])
plt.plot(range(kk-2),mat_dealamount[1:kk-1],'r.')
for i in range(0,kk-2,int((kk-2)/5)):
    plt.text(i, mat_dealamount[i]*0.8, mat_timetag[i,1].decode('big5'))
    plt.text(i, mat_dealamount[i]*0.7-10, mat_timetag[i,0].decode('big5'))
plt.legend(['deal amount'])
plt.savefig("html/Figure/"+str(num)+"_"+datestr[2:]+".png")
print("html/Figure/"+str(num)+"_"+datestr[2:]+".png")