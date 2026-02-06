# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:21:58 2021
@author: rubby
"""
from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen
from urllib.parse import urlparse
import datetime
import random
import re
import time
import sys
import json
import numpy as np
from newslib import *
import pandas as pd
import twstock

url = 'https://www.tpex.org.tw/web/bond/publish/convertible_bond_search/memo.php?l=zh-tw'
bs = getPage(url)

mat = bs.table;
nodes = mat.find_all('td')

strtime=re.split('-| |:|\.',get_datetime())
datestr = strtime[0]+strtime[1]+strtime[2]

# 下載股價
dict_stock = read_stock_fulllist("stock_list.txt")
fulltext=str_tr(str_td("  代號  ")+str_td("  公司  ")+str_td("  到期月  ")+str_td("  到期年  ")\
                +str_td("  現價  ")+str_td("  發行價格  ")\
                    +str_td("  EPS  ")+str_td("  毛利率  "))

dict_wishlist={}
tdi = 0
for line in nodes:
    if tdi%8 == 0:
        num=int(line.string)
        name = nodes[tdi+1].string
        fullname = nodes[tdi+2].string
        bgntime = nodes[tdi+3].string
        endtime = nodes[tdi+4].string
        issueamount = nodes[tdi+6].string
        tmp_dict=nodes[tdi+7].a
        website = tmp_dict['href']
        
        mattime0 = re.split('/',bgntime)
        bgnyy = int(mattime0[0])
        bgnmm = int(mattime0[1])
        bgndd = int(mattime0[2])
        
        mattime1 = re.split('/',endtime)
        endyy = int(mattime1[0])
        endmm = int(mattime1[1])
        enddd = int(mattime1[2])
        if (bgnyy==2021 and bgnmm>=3):
            #df, data,price,dealnum = read_stock_daily(str(int(num/10)), datestr)
            price, dealnum, EPS, delEPS, netrate = getGoodInfo(int(num/10))
            if price == -999:
                num2=[value for key, value in dict_stock.items() if key in name]
                #df, data,price,dealnum = read_stock_daily(int(num2[0]), datestr)
                price, dealnum, EPS, delEPS, netrate = getGoodInfo(int(num2[0]))
            bs3, body, current_price, rate = scrapeBondInfo(website)
            fulltext = fulltext + str_tr(str_td(str(num))+str_td(name)+str_td(\
                str(endmm))+str_td(str(endyy))+str_td(str(price))+str_td(str(\
                current_price))+str_td(str(EPS))+str_td(str(netrate)))
            print(num, name, endmm, endyy, price, current_price, dealnum)
            data = get_full_year_data(int(num/10),price,current_price)
            print(df)
            dict_wishlist[name]=num
            time.sleep(random.random()*3)
    tdi = tdi + 1

#Torch
filename = 'html/Bond_'+datestr+'.html'
fo = open(filename,'w')  

f = open('html/Sample.html')
ii = 0
lines = f.readlines()


lines2 = ''
for line in lines:
    if 'SampleTitleXXX' in line:
        lines2=lines2 + '可轉債'
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

lines2 = re.sub('SampleHeaderXXXXX',"轉換公司債發行資料",lines2);
lines2 = re.sub('SampleSubHeaderXXXXX',datestr,lines2);
lines2 = re.sub('SampleClassXXXXX',datestr,lines2);

#lines2 = re.sub('SamplePNG',num,lines2);
for line in lines2:
    fo.write(line)
fo.close()