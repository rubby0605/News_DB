#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 07:35:04 2021

@author: rubylintu
"""
import re
filename = "stock_list.txt"
f = open(filename,'r')
lines = f.readlines()
dict_stock = {}
for line in lines:
    mat = re.split(' |\t|\n',line);
    if len(mat)<=2:
        continue
    else:
        dict_stock[mat[2]] = mat[0]
        print(mat)
        
