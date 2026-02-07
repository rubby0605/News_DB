#!/bin/bash
input="stock_list_less.txt"

./arg_info_stock.sh `cat stock_list_less.txt | awk '{print $1}'` > tmp_cmd5.sh
# ./arg_plot.sh `cat stock_list_less.txt | awk '{print $1}'` >> tmp_cmd5.sh

chmod u+x tmp_cmd5.sh
./tmp_cmd5.sh

  
