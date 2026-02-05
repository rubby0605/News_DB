#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 23:48:34 2021

@author: rubylintu
"""
import os,sys
from newslib import *
import time


dict_stock = read_stock_list('stock_list_less.txt')
fulltext=''

for stock in dict_stock.keys():
    stock_number = int(dict_stock[stock])
    value = get_index_wd_onemonth(stock_number)
    time.sleep(10*random.random())
    fulltext = fulltext + str_tr(str_td(stock) + str_td(str(stock_number)) + str_td(str(value)))

#Torch
datemat = re.split('\-| |\:|\.',str(datetime.datetime.today()))
datestr = datemat[0]+datemat[1]+datemat[2]
filename = 'html/WD_'+datestr+'.html'
fo = open(filename,'w')  

f = open('html/Sample.html')
ii = 0
lines = f.readlines()

lines2 = ''
for line in lines:
    if 'SampleTitleXXX' in line:
        lines2=lines2 + '洗盤指數'
        print('1')
    elif '.mp3' in line or '.png' in line:
        line=''
    elif 'FullText' in line:
        lines2 = lines2 + "<table>"
        for sentence in fulltext:
            lines2 = lines2 + sentence
        lines2 = lines2 + "</table>" 
        print('!!!')
    else:
        lines2=lines2 + line
    ii = ii + 1

lines2 = re.sub('SampleHeaderXXXXX',"洗盤指數計算",lines2);
lines2 = re.sub('SampleSubHeaderXXXXX',datestr,lines2);
lines2 = re.sub('SampleClassXXXXX',datestr,lines2);

#lines2 = re.sub('SamplePNG',num,lines2);
for line in lines2:
    fo.write(line)
fo.close()


