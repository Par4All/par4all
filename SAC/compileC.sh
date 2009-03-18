#!/bin/sh
set -e

SCRIPTNAME="`basename $0`"

perror()
{
    echo "error: $1" 1>&2
    exit
}

test "$#" -gt 1 || perror "not enough args, usage: $SCRIPTNAME project module [outfile] "
PROJECT_NAME="$1"
MODULE_NAME="$2"
OUTFILE=/dev/stdout

if test -n "$3"; then
    OUTFILE="$3"
fi

TARGET=$PROJECT_NAME.database/$MODULE_NAME/$MODULE_NAME.pref

{
    cat include/sse.h
    sed -r  -e 's/float v4sf_([^ ,]+)\[.*\]/__m128 \1/g' \
            -e 's/v4sf_([^ ,]+)/\1/g' $TARGET
} > $OUTFILE

cat 1>&2 << EOF
********************************************
substitution done,
output can be compiled using
gcc -march=native -O3 -c $OUTFILE
********************************************
EOF

