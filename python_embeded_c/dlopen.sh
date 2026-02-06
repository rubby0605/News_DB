#/bin/bash

#build f1.so
gcc -c f1.c -o f1.o -fPIC 
gcc f1.o -o f1.so -shared

#build main
gcc -c main_dlopen.c -o main.o
gcc main.o -o test -ldl
