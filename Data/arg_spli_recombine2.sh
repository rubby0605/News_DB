
#!/bin/bash
# usage : ./arg_spl_recombine.sh 1 2 3 4 5
# output will be
# 1 2 3 
# 1 2 4
# 1 2 5


i=1;

for ARG
do
	if [ ${i} -eq 1 ]
	then
		((i=i+1))
		str1=$ARG;
		continue;
	elif [ ${i} -eq 2 ]
	then
		((i=i+1))
		str2=$ARG;
		continue;
	fi
	echo $str1 $str2 $ARG
done

