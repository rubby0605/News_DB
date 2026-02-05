#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 20:00:38 2021

@author: rubylintu
"""

from newslib import *

        
columns = ['c','n','z','tv','v','o','h','l','y']
dict_stock = read_stock_list('stock_list_less.txt') #input file 
stock_list_str = dict_stock.keys()
stock_list = [int(dict_stock[stock]) for stock in stock_list_str]

fi = open('Data/trace_stock_DB.txt','a')##############

j = 1

while True:
    data = craw_realtime(stock_list)
    for i in range(len(dict_stock)-1):
        line = ''
        for column in columns:
            value = data['msgArray'][i][column]
            line = line + '\t' + value
        line = line + str(datetime.datetime.now()) + '\n'
        fi.write(line)
    time.sleep(15*random.random())
    print('run'+str(j))
    j= j +1