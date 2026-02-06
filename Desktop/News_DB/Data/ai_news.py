#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 19:25:44 2021

@author: rubylintu
"""
from newslib import *
import random
import torch
from torchtext import data
import gensim
import torch.nn as nn
import numpy as np
from gensim import corpora
from gensim import models
from gensim.models import LdaModel
from gensim.models import TfidfModel
from gensim.models import CoherenceModel

keyword=sys.argv[1]
# parameter setting
pages=set()
random.seed(datetime.datetime.now())
content_dict={}
#keyword = sys.argv[1]
print(keyword)
#initialize


# Main

str_now= get_datetime()
url, title, fulltext = scrapBingNews(keyword)
url, title, fulltext2 = scrapGoogleNews(keyword)
fulltext = fulltext + fulltext2
str_full_text, content_dict = getNewsDataClean(fulltext, content_dict, str_now)

#Torch
filename = 'Data/dict_'+keyword+'.csv'
f = open(filename,'r')  
Lines=f.readlines()
dict_word = {}
result = list()  
mat_dict=np.zeros([100,3]) #####################change to how many elements
num=0
for line in Lines[2:]:  
    A = line.split(',')
    dict_word[A[0]] = num
    mat_dict[num,0:2] = A[1:3]
    num = num+1
f.close() 
print(dict_word)

num_dict = len(dict_word)





