#!/bin/bash
#combine 2 csv files into 1 new

file1=$1  #output file name
file2=$2  #file1
file3=$3  #file2


cat $file2 >> $file1
cat $file3 >> $file1

echo "cat $file2 >> $file1"
echo "cat $file3 >> $file1"

echo "done"
