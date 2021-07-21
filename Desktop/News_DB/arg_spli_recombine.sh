#!/bin/bash
# usage : ./arg_spl_recombine.sh 1 2 3 4
# output will be
# 1 2
# 1 3
# 1 4


i=1;

for ARG
do
	if [ ${i} -eq 1 ]
	then
		i=0
		str1=$ARG;
		continue;
	fi
	echo $str1 $ARG
done

