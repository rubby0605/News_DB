#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 11:46:23 2022

@author: rubylintu
"""
import math
import random
import os
from newslib import *
import twstock
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
import pandas as pd

#----------------------------------------------------------------

url='https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv'
url='https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv'
dfweb=pd.read_csv(url)
[td,fd]=dfweb.shape
X = np.zeros([td, 2])
pushID= np.zeros([td, 5])
df2 = pd.DataFrame(pd.array(["000" for i in range(td)], dtype="string"))
df_name = pd.DataFrame(pd.array(["000" for i in range(10000)], dtype="string"))
df_pushName = pd.DataFrame(pd.array([["000" for i in range(5)] for j in range(td)], dtype="string"))
ID = np.zeros([td, 1])
    
filename = 'capital2.csv'
df = pd.read_csv(filename)
dict_csv = dict()
for i in range(10000):
    dict_csv[df.iloc[i,1]] = i

dict_ind = dict()
for i in range(td):
    ID[i,0] = dfweb.iloc[i,1]
    X[i,0] = dfweb.iloc[i,17]
    df2.iloc[i,0] = dfweb.iloc[i,3]
    if dfweb.iloc[i,1] < 10000:
        df_name.iloc[dfweb.iloc[i,1],0] = dfweb.iloc[i,3]
    print(df2.iloc[i,0])
    #str_ind = dfweb.iloc[i,5]
    #if str_ind in dict_ind:
    #    num_ind = dict_ind[str_ind]
    #else:
    #    dict_ind[str_ind] = np.size(dict_ind.keys())
    #    num_ind = dict_ind[str_ind]
    #X[i,1] = dict_ind[str_ind]
    if dfweb.iloc[i,1] in dict_csv:
        i0 = dict_csv[dfweb.iloc[i,1]]
        X[i,1] = df.iloc[i0, 2]
    else:
        X[i,1] = -999
        print(i)

nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

for i in range(td):
    for j in range(5):
        if ID[indices[i,j],0] < 10000:
            df_pushName.iloc[i, j] = df_name.iloc[int(ID[indices[i,j],0]),0] + str(int(ID[indices[i,j],0]))


has_cal = np.zeros([td])
for i in range(td):
    if has_cal[i] == 1:
        continue
    X1 = [X[j,0] for j in range(td) if X[j,1] == X[i,1]]
    X2 = [X[j,1] for j in range(td) if X[j,1] == X[i,1]]
    id1 = [j for j in range(td) if X[j,1] == X[i,1]]
    
    print(df_name.iloc[[ID[id1[i],0] for i in range(np.size(X1)-1)],0])
    X3 = np.concatenate([[X1, X2]], axis=1).transpose()
    if np.size(X1) == 1:
        distances1, indices1 = 0, i
    elif np.size(X1) <=5:
        nbrs = NearestNeighbors(n_neighbors=np.size(X1)-1, algorithm='ball_tree').fit(X3)
        distances1, indices1 = nbrs.kneighbors(X3)
        for j in range(np.size(X1)-1):
            X4 = df_name.iloc[[ID[id1[b+1], 0] if ID[id1[b+1], 0] < 10000 and indices1[b,j] == 1 else 9999 for b in range(np.size(X1)-1)],0]
            df_pushName.iloc[i,j] = [ele for ele in X4 if '000' not in X4]
    else:
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X3)
        distances1, indices1 = nbrs.kneighbors(X3)
        for j in range(5):
            df_pushName.iloc[i,j] = df_name.iloc[[ID[id1[b+1], 0] if ID[id1[b+1], 0] < 10000 and indices1[b,j] == 1 else 9999 for b in range(np.size(X1)-1)],0]
    
df0 = pd.DataFrame(X)
df1 = pd.DataFrame(ID)
df3 = pd.DataFrame(pushID)
df_op = pd.concat([df0, df1, df2, df_pushName], axis=1)
df_op.to_csv('stock_classify.csv')

    
    
    
    