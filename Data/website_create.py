#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 18:49:20 2021

@author: rubylintu
"""

from newslib import *
import random

keyword=sys.argv[1]
ekeyword = sys.argv[2]
# parameter setting
pages=set()
random.seed(datetime.datetime.now())
content_dict={}
print(keyword)
#initialize


# Main

str_now= get_datetime()
url, title, fulltext = scrapBingNews(keyword)
url, title, fulltext2 = scrapGoogleNews(keyword)
fulltext = fulltext + fulltext2
str_full_text, content_dict = getNewsDataClean(fulltext, content_dict, str_now)

#Torch
filename = 'html/News_'+ekeyword+'.html'
fo = open(filename,'w')  

f = open('html/Sample.html')
ii = 0
lines = f.readlines()


lines2 = ''
for line in lines:
    if 'SampleTitleXXX' in line:
        lines2=lines2 + keyword
        print('1')
    elif 'FullText' in line:
        for sentence in content_dict.keys():
            lines2 = lines2 + sentence
        print('!!!')
    else:
        lines2=lines2 + line
    ii = ii + 1
    
lines2 = re.sub('SampleHeaderXXXXX',keyword,lines2);
lines2 = re.sub('SampleSubHeaderXXXXX',str_now,lines2);


lines2 = re.sub('SampleClassXXXXX',keyword+str_now,lines2);
lines2 = re.sub('SampleMp3XXXXX',ekeyword,lines2);
#lines2 = re.sub('SamplePNG',num,lines2);


dict_bull = read_keyword('bull')
dict_bear = read_keyword('bear')
add_keyword(keyword)

for word in dict_bull.keys():
    print(word)
    lines2 = highlight_word(lines2, word)
    
for word in dict_bear.keys():
    lines2 = lowlight_word(lines2, word)

lines2 = lowlight_word(lines2, '不適用')
    
lines2 = highlight_word(lines2, '高階')
lines2 = highlight_word(lines2, '第一')
lines2 = highlight_word(lines2, '年增')
lines2 = highlight_word(lines2, '成長')
lines2 = highlight_word(lines2, '上升')

for line in lines2:
    fo.write(line)
fo.close()
