#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:02:06 2022

@author: rubylintu
"""

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
import twstock
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

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
        
        

dict_stock = read_stock_fulllist_id0('stock_list.txt')
a1 = np.zeros([10000,1])
a2 = np.zeros([10000,1])

 
# Create DataFrame
df0 = pd.DataFrame(a1)
df1 = pd.DataFrame(a2)
df2 = pd.DataFrame(pd.array(["000" for i in range(10000)], dtype="string"))
df=pd.concat([df0,df1,df2],axis=1)


#----------------------------------------------------------------

n=0
for num in dict_stock.keys():
    if int(num) < 1534:
        continue
    if dict_stock[num] == 1:
        continue
    capital, industry = getGoodInfo2(num)
    df.iloc[0,0] = num
    df.iloc[0,1] = capital
    df.iloc[0,2] = industry
    dict_stock[num] = 1
    n+=1
    time.sleep(0.3)
    if int(n/300)*300 == n:
        df.to_csv("capital.csv")
    print(num, capital, industry)
    

#----------------------------------------------------------------
url='https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv'
dfweb=pd.read_csv(url)
[td,fd]=dfweb.shape
X = np.zeros([td, 2])
ID = np.zeros([td, 1])
dict_ind = dict()
for i in range(td):
    ID[i,0] = dfweb.iloc[i,1]
    X[i,0] = dfweb.iloc[i,17]
    #str_ind = dfweb.iloc[i,5]
    #if str_ind in dict_ind:
    #    num_ind = dict_ind[str_ind]
    #else:
    #    dict_ind[str_ind] = np.size(dict_ind.keys())
    #    num_ind = dict_ind[str_ind]
    #X[i,1] = dict_ind[str_ind]
    X[i,1] = dfweb.iloc[i,5]
has_cal = np.zeros([td])
for i in range(td):
    if has_cal[i] == 1:
        continue
    X1 = [X[j,0] for j in range(td) if X[j,1] == X[i,1]]
    id1 = [j for j in range(td) if X[j,1] == X[i,1]]
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X1)
    distances, indices = nbrs.kneighbors(X1)
#nbrs.kneighbors_graph(X).toarray()






