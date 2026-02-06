#!/bin/bash
# this file will output series of command to
# use python to control the news web scraping 
# wish list will perform like following
# `cat Data/wishlist.txt` >> 台郡 聯發科 元大 國泰金 鴻海
#
for i in {1..100}
do
	rm tmp_cmd.sh
	./arg_py1.sh `cat wishlist.txt` |sed -e 's/_/ /g' >> tmp_cmd1.sh
	./arg_py2.sh `cat wishlist.txt` |sed -e 's/_/ /g' >> tmp_cmd2.sh
	#./arg_spli_recombine2.sh python KeyWordDemo.py `cat wishlist.txt |sed 's/^[0-9]*//g'` | sed -e 's/_/ /g' >> tmp_cmd1.sh
	#./arg_spli_recombine2.sh python website_create.py `cat wishlist.txt |sed 's/^[0-9]*//g'` | sed -e 's/_/ /g' >> tmp_cmd1.sh
	#chmod u+x tmp_cmd.sh 
	chmod u+x tmp_cmd1.sh 
	chmod u+x tmp_cmd2.sh 
	#./tmp_cmd.sh &
	./tmp_cmd1.sh &
	./tmp_cmd2.sh &
	sleep 360
done

