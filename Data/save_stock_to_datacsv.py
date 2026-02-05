#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 23:31:07 2022

@author: rubylintu
"""
import time
from newslib import *
dict_stock = read_stock_list('stock_list_less.txt') #input file 
stock_list_str = dict_stock.keys()
f = open("Data/stock_data.csv",'w')
f.write('Name,price,EPS,delEPS,netrate\n')
i=0
for stock in stock_list_str:
    num = dict_stock[stock]
    price, dealnum, EPS, delEPS, netrate = getGoodInfo(num)
    f.write(stock+','+str(price)+','+str(EPS)+','+str(delEPS)+','+str(netrate)+'\n')
    i=i+1
    time.sleep(0.1)
    print(i)
f.close()
