#!/bin/bash
#usage : ./listfiles.sh sh output.csv
key=$1
outputfile=$2

for VAL in $(echo ""*.${key}"")
do
	echo "cat " $VAL " >> " $outputfile
done

