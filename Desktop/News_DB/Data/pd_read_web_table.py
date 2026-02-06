#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:02:48 2022

@author: rubylintu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unicodedata import normalize
from bs4 import BeautifulSoup

filename='/Users/rubylintu/Desktop/News_DB/Data/capital/capital2.html'

fp = open(filename,'rb')
bs = BeautifulSoup(fp, 'html.parser')
a1 = np.zeros([10000,1])
a2 = np.zeros([10000,1])
a3 = np.zeros([10000,1])

# Create DataFrame
df0 = pd.DataFrame(a1)
df1 = pd.DataFrame(a2)
df2 = pd.DataFrame(a3)
df=pd.concat([df0,df1,df2],axis=1)

bs1 = bs.find('table').tr
for i in range(10000):
    if bs1.find_next('tr') is not None:
        bs1 = bs1.find_next('tr')
    else:
        break
    bs2 = bs1
    for n in range(11):
        if n == 1:
            df.iloc[i,0] = bs2.text
        elif n == 4:
            df.iloc[i,2] = bs2.text
        elif n == 7:
            df.iloc[i,1] = bs2.text
        print(bs2)
        bs2 = bs2.find_next('td')
    print(i)
fp.close()
df.to_csv('capital2.csv')