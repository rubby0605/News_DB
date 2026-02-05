#!/usr/lib/python2.7
#coding: utf-8

from newslib import *
import random
keyword=sys.argv[1]


# parameter setting
pages=set()
random.seed(datetime.datetime.now())
basepath='/Users/rubylintu/Desktop/News_DB/Data/Data/'
content_dict={}
#keyword = sys.argv[1]
print(keyword)
#initialize


# Main

str_now= get_datetime()
url, title, fulltext, bs = scrapBingNews(keyword)
url, title, fulltext2, bs2 = scrapGoogleNews(keyword)
fulltext = fulltext + fulltext2
str_full_text, content_dict = getNewsDate(fulltext2, content_dict, str_now, bs2)
#return_val = save_csv_file(str_full_text, content_dict, basepath, str_now, keyword)
return_val = save_files(str_full_text, content_dict, basepath, str_now, keyword)
print(return_val)

