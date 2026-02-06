#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 07:35:04 2021

@author: rubylintu
"""

#import twstock
import re
import os, sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
from newslib import *
import time
import random


def scrapeHiStockSamp(num): 
    url = 'https://histock.tw/stock/' + str(num)  
    response = requests.get(url)
    #print(response.text)
    root=BeautifulSoup(response.text, "html.parser")
    deal=root.find("span", class_="clr-rd")
    return url, root, deal


dict_stock = read_stock_list("stock_list_less.txt")
for key in dict_stock.keys():
    num = dict_stock[key]
    #url, title, data, bs = get_stock_info(num)
    #deal, stockprice = scrapeGoogleStockInfo(num)
    #print(key, num, time, stockprice, deal)
    url, data = get_stock_info(num)
    strtime = datetime.datetime.now()  
    print(key, num, strtime)
    time.sleep(random.random()*30)
    print(data)

#url = 'https://histock.tw/stock/warrantstats.aspx'
#dict_stock = read_stock_list('stock_list.txt')
#for stock in dict_stock.keys():
#    numstock = dict_stock[stock]
#    passstock = twstock.realtime.get(numstock)
#    print(str(passstock)+stock)

    
#def write_json(new_data, filename='data.json'):
#    with open(filename,'r+') as file:
#          # First we load existing data into a dict.
#        file_data = json.load(file)
#        # Join new_data with file_data inside emp_details
#        file_data["emp_details"].append(new_data)
#        # Sets file's current position at offset.
#        file.seek(0)
#        # convert back to json.
#        json.dump(file_data, file, indent = 4)
