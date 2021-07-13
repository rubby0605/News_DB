#!/bin/bash
# this file will output series of command to
# use python to control the news web scraping 
# wish list will perform like following
# `cat Data/wishlist.txt` >> 台郡 聯發科 元大 國泰金 鴻海
#
./argsum.sh `cat Data/wishlist.txt` > wishcount_tmp.tmp

echo "sleep 1; " > cmdline.sh
chmod u+x cmdline.sh
while read -r line;
do
		if (( `echo ${#line}` >= 1 ))
		then
			echo "python scrp_news_copy.py $line &" >> cmdline.sh
		fi
done < Data/wishlist.txt
#echo "sleep 60" >> cmdline.sh
#exe
./cmdline.sh
sleep 10;
rm cmdline.sh

# for i in $@
# 	echo "python scrp_new_copy.py $i "
# done

