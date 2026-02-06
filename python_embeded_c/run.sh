#/bin/bash

#build f1.so
gcc -c f1.c -o f1.o -fPIC 
gcc f1.o -o f1.so -shared

#build main
gcc -c main.c -o main.o
gcc main.o /Users/rubylintu/python_embeded_c/f1.so -o test 
