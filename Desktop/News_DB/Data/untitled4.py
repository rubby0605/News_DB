#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:16:30 2024

@author: rubylintu
"""

import sys
from newslib import *

import json
import random
import csv
import numpy as np
import ollama

keyword="台積電"
ekeyword=123
#initialize
pages=set()
random.seed(datetime.datetime.now())
content_dict={}
print(keyword)

# Main

str_now= get_datetime()
mat1 = scrapBingNews(keyword)
mat2 = scrapGoogleNews(keyword)
url1 = mat1[0]
title1 = mat1[1] 
fulltext1 = mat1[2]
url2 = mat2[0]
title2 = mat2[1] 
fulltext2 = mat2[2]

fulltext = fulltext1 + fulltext2


question = "請問以下新聞是否影響股價,請回答兩字'正面'或是'反面'"


response = ollama.chat(
    model="llama2",
    messages = [
        {
            "role": "user", 
            "content": question+fulltext
        }
    ]
)

print(response)