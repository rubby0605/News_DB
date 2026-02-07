#!/bin/bash 
# argument split line by line
# usage : ./argsplt.sh 1 2 3 

for ARG
do
# echo "grep $ARG Data/DB_stock  > Data/${ARG}_`date +%y%m%d`.txt"
echo "grep $ARG Data/trace_stock_DB.txt > Data/${ARG}_`date +%y%m%d`.txt"


done
