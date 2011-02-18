#!/bin/sh

python_inter=`pkg-config --variable python pips`

for f in `ls *.c`
do
    fname=`basename $f .c`
    outfile=py-$fname.py
    echo "from __future__ import with_statement" > $outfile
    echo "from pyps import *" >> $outfile
    echo "import openmp" >> $outfile
    echo "" >> $outfile
    echo "with workspace(\"$f\") as w:" >> $outfile
    echo -e "\tw.all_functions.openmp(verbose=True)" >> $outfile
    mkdir -p py-$fname.result
    $python_inter $outfile > py-$fname.result/test
done
