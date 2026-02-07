#!/bin/bash
#usage : ./listfiles.sh sh output.cssv
key=$1

#for VAL in $(find -type f -name `echo $ARG` ) {0..5}
#for VAL in $(find . -type f -name `echo $ARG`)
for VAL in $(ls *.`echo ${key}`)
do
        echo "cat " $VAL " >> " $outputfile > tmp.sh
done

##!/bin/bash
##usage : ./listfiles.sh sh output.cssv
#key=$1
#
#for VAL in $(echo ""*.${key}"")
#do
#        echo "cat " $VAL " >> " $outputfile
#done
#


