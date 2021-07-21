#!/bin/bash
# this file will output series of command to
# use python to control the news web scraping 
# wish list will perform like following
# `cat Data/wishlist.txt` >> 台郡 聯發科 元大 國泰金 鴻海
#
./arg_spli_recombine.sh ./scrp_news_copy.py `cat wishlist.txt |sed 's/^[0-9]*//g'` 
echo "sleep 60" >> cmdline.sh

