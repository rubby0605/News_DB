#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# FileName: KeyWordDemo.py
# Developer: Trueming (trueming@gmail.com)
import sys
from newslib import *

import json
import random
import csv
import numpy as np


keyword=sys.argv[1]
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

str_full_text, content_dict = getNewsDataClean(fulltext, content_dict, str_now)
save_mp3(str_full_text, content_dict, ekeyword)
#return_val = save_csv_file(str_full_text, content_dict, basepath, str_now, keyword)
#return_val = save_files(str_full_text, content_dict, basepath, str_now, keyword)
#print(return_val)

# read dictionary


if __name__ == "__main__":

    try:
        #使用自己的斷詞額度。
        with open("../../account.info", "r") as f:
            userDICT = json.loads(f.read())
        username = userDICT["email"]
        apikey = userDICT["apikey"]
        atc = Articut(username=userDICT["email"], apikey=userDICT["apikey"])
    except:
        #使用免費的斷詞額度。
        #實體化 Articut()
        atc = Articut()

    # 載入 Demo 用的文字
    #text = open(sys.argv[1], "r").read()
    #sentLIST = content_dict.keys

    # 載入 Dictionary 原本的詞彙
try:
    filename  = "Data/dict_" + keyword + ".csv"
    f = open(filename,'r')  
    Lines=f.readlines()
    mat_dict=np.zeros([500,3]) #####################change to how many elements
    word_dict = {}
    result = list()  
    for line in Lines[2:]:  
        wordList = line.split(',')
        for word in wordList:
            word_dict[word] = 1
    print(word_dict)
except IOError:
    print('File is not accessible')
    f = open(filename,'w')  
    word_dict = {}
    f.close


print("ArticutAPI Term Extraction Demo")
for sentence in content_dict.keys():
	if "" == sentence.strip():
		continue
	result = atc.parse(sentence)
	if result["status"]:
		print("{}\nInput: {}".format('#'*20 , sentence))
		# TextRank 抽取句子關鍵詞並排序
		wordLIST = atc.analyse.textrank(result)
		print(wordLIST)
		# TFIDF 抽取句子關鍵詞
		wordLIST = atc.analyse.extract_tags(result)
		for word in wordLIST:
		 if(word not in word_dict):
		  print(word)
		  word_dict[word] = 1
		print("TF-IDF:", wordLIST)

#save dictionary

str_now2 = re.sub(r'[-|: ]','_',ekeyword)
str_now3 = re.sub(r'_[0-9]+_[0-9]+.[0-9]+$','00',str_now2)
str_now = str_now3
#save csv file as $date.csv
csv_file_name = filename
print(csv_file_name)
num=0
with open(csv_file_name, 'w') as f:
	for key in word_dict:
		f.write("%s,%s"%(key,word_dict[key]))
		num = num + 1
		if(num%10 == 0):
			f.write("\n")
		else:
			f.write(",")
	f.close()
print('Done')
