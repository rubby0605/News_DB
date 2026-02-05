#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 05:52:50 2022

@author: rubylintu
"""
from newslib import *

#"日期","成交股數","成交金額","開盤價","最高價","最低價","收盤價","漲跌價差","成交筆數"

dict_stock = read_stock_list('stock_list_less.txt') #input file 
stock_list_str = dict_stock.keys()
stock_list = [int(dict_stock[stock]) for stock in stock_list_str]
for num in stock_list:
    df = get_index_wd_fullmonth(num)
    if not np.size(df) == 1:
        fig = plot_candles(df, title=str(num))
        fig.savefig('KL/' + str(num) + '.png')
    


