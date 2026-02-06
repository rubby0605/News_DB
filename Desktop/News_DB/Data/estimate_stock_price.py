#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 13:00:50 2022

@author: rubylintu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 10:10:13 2022

@author: rubylintu
"""
import math
import random
import os
from newslib import *

class stock:
    def __init__(self,price, amount):
        self.price = price
        self.amount = amount

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
SGM_X = 5 # average deal earning
SGM_Y = 2 # deal amount per person

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 20:00:38 2021

@author: rubylintu
"""

        
columns = ['c','n','z','tv','v','o','h','l','y']
dict_stock = read_stock_list('stock_list_less.txt') #input file 
stock_list_str = dict_stock.keys()
stock_list = [int(dict_stock[stock]) for stock in stock_list_str]

datestr_list = ['220422']

for stock in [ '台積電']:#dict_stock.keys():
    plt.figure(figsize=((10,8)))
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    num = dict_stock[stock]
    total = 0
    amount = 0
    pre_price = 0
    oldmat4 = 0
    init_i = datetime.timedelta(0)
    deltatt = 0
    dateid=0
    for datestr in datestr_list:
        f = open(str(num)+'_'+datestr + '.txt')
        print(total)
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
                if i!=0:
                    oldbuffer2 = digest_oldbuffer(oldbuffer, price, 0.2)
                    oldbuffer = np.zeros([10000,1])
                if i == 0:
                    buffer = np.zeros(dn)
                    oldbuffer = np.zeros([10000,1])
                if dn != 0:
                    buffer = np.zeros(dn)
                    for num_trace in range(dn):
                       aa= particle(price,0,0, 1)
                       aa.get_2dnpdf(MUX, MUY, SGM_X, SGM_Y)
                       buffer[num_trace] = aa.bp+aa.xx
                oldbuffer = combine_buffers(buffer,oldbuffer)
                plt.hist(oldbuffer)
                print(dn)
                amount = amount + dn
                total = total + price * float(dn)
            #total = total * decay
            #amount = amount * decay
            b= datetime.datetime(2022, int(mm), int(dd), int(mat[11]), int(mat[12]), int(float(mat[13])))
            time_delta = b - a
            tt = time_delta.total_seconds() / 5
            StopIteration
            print(amount)
            plt.plot(tt+deltatt, price,"k.")
            if amount !=0:
                plt.plot(tt+deltatt, total/amount,"r.")
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










