#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 20:00:38 2021

@author: rubylintu
"""

from newslib import *

class stock:
    def __init__(self,price, amount):
        self.price = price
        self.amount = amount
        
columns = ['c','n','z','tv','v','o','h','l','y']
dict_stock = read_stock_list('stock_list_less.txt') #input file 
stock_list_str = dict_stock.keys()
stock_list = [int(dict_stock[stock]) for stock in stock_list_str]

datestr_list = ['220113']
decay1 = 0.99
decay2 = 0.995
decay3 = 0.999
for stock in dict_stock.keys():
    plt.figure(figsize=((10,8)))
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    num = dict_stock[stock]
    total1 = 0
    total2 = 0
    total3 = 0
    amount1 = 0
    amount2 = 0
    amount3 = 0
    pre_price = 0
    oldmat4 = 0
    init_i = datetime.timedelta(0)
    deltatt = 0
    dateid=0
    for datestr in datestr_list:
        f = open('Data/' + str(num)+'_'+datestr + '.txt')
        print(total3)
        lines = f.readlines()
        i=0
        for line in lines:
            mat = re.split('\t|2022-| |:|\n', line)
            if int(num) != int(mat[1]):
                continue
            [mm,dd]=re.split('-',mat[10])
            if i == 0:
                a = datetime.datetime(2022, int(mm), int(dd), int(mat[11]), int(mat[12]), int(float(mat[13])))
            if '-' not in mat[3]:
                price = float(mat[3])
            else:
                if pre_price == 0:
                    if '-' not in mat[6]:
                        price = float(mat[6])
                    else:
                        price = 0
                else:
                    price = pre_price
            if "-" in mat[4] or price==0:
                dn = 0
            else:
                if mat[4] == oldmat4:
                    dn = 0
                else:
                    dn = int(mat[4])
            if price != 0:
                amount1 = amount1 + dn
                amount2 = amount2 + dn
                amount3 = amount3 + dn
                total1 = total1 + price * float(dn)
                total2 = total2 + price * float(dn)
                total3 = total3 + price * float(dn)
            total1 = total1 * decay1
            total2 = total2 * decay2
            total3 = total3 * decay3
            amount1 = amount1 * decay1
            amount2 = amount2 * decay2
            amount3 = amount3 * decay3
            b= datetime.datetime(2022, int(mm), int(dd), int(mat[11]), int(mat[12]), int(float(mat[13])))
            time_delta = b - a
            tt = time_delta.total_seconds() / 5
            plt.plot(tt+deltatt, price,"k.")
            if amount1 !=0:
                plt.plot(tt+deltatt, total1/amount1,"r.")
                plt.plot(tt+deltatt, total2/amount2,"g.")
                plt.plot(tt+deltatt, total3/amount3,"b.")
            if '-' in mat[4]:
                oldmat4 = 0
            else:
                oldmat4 = mat[4]
            pre_price = price
            i = i + 1
        deltatt = deltatt + 4.5*3600 / 5
    #plt.axis([1,i,total/(1+amount)*0.8,total/(1+amount)*1.2])
    plt.title(mat[1])
    plt.savefig("html/" + num + ".png")
    StopIteration
   #b = datetime.datetime(2017, 5, 16, 8, 21, 10)
    
    
    #          ['c',   'n',       'z',      'tv' ,         'v',      'o',    'h',    'l',    'y'] 
    #分別代表 ['股票代號','公司簡稱','當盤成交價','當盤成交量','累積成交量','開盤價','最高價','最低價','昨收價']

"""
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
"""