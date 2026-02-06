#!/bin/bash
#dirs="dir1 dir2 dir3"
dirs="."
find_depth="-maxdepth 1"
rm -f cscope.files
for dir in $dirs; do
	echo "search $dir ..."
	find $dir $find_depth -name "*.py" >> cscope.files
done

sort -u cscope.files > cscope.files.unique
mv cscope.files.unique cscope.files

echo "create cscope DB ..."
cscope -Rbqk -i cscope.files

# cscope.out

