#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 13:23:31 2021

@author: rubylintu
"""

import csv
import sys, os
from os import listdir
from os.path import isfile, isdir, join
import re

# 指定要列出所有檔案的目錄
keyword = '元大'
mypath = "Data/" + keyword +'/'
outputfilename = mypath + keyword + 'full.csv'
csv_dict={}

def arange_files(mypath):
    # 取得所有檔案與子目錄名稱
    files = listdir(mypath)
    print(len(files))
    #initialize
    csv_dict={}
    mp3_dict={}

#    以迴圈處理
    for f in files:
        if ( '.csv' in str(f)):
            csv_dict[f] = 1
        elif( '.mp3' in f):
            mp3_dict[f] = 2
        else:
            print("x", f)
    #outputfile = open(f, mode='w')
    return csv_dict, mp3_dict
    
#find out csv files
#for keys in file_dict:
#    if file_dict[keys] == 2:
#        print(keys)


def get_datetime():
    str_now = str(datetime.datetime.now())
    return str_now

def getDictFromCsv(filename, outputfile, mydict):
    print(filename)
    i=0
    with open(filename, mode='r') as infile:
        reader = csv.reader(infile)
        writer = csv.writer(outputfile)
        for row in reader:
            if(len(row) == 2):
                k, v = row
                mydict[k] = v
    return mydict

#content_dict = getDictFromCsv(filename)
#print( content_dict )

csv_dict, mp3_dict = arange_files(mypath)
for filename in csv_dict:
    with open(outputfilename) as outputfile:
        print(mypath+filename)
        csv_dict = getDictFromCsv(mypath+filename, outputfile, csv_dict)
        print(csv_dict)
        print(outputfilename)

