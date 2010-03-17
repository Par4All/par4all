#!/bin/bash
echo $SHELL

cd Preprocessor
for f in *.c
do
    echo $f
    name=`basename $f .c`
    echo $name
    if [  -e $name.tpips ]
    then 
	echo $name.tpips exists
    else
	echo $name.tpips does not exist
#	gcc -E $name.c >$name.e
#       Comments seem to be lost with cpp
	cpp -P -C $name.c >$name.e
	mv $name.c $name.i
	mv $name.e $name.c
    fi
done
