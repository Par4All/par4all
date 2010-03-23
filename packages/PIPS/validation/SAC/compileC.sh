#!/bin/sh
set -e

SCRIPTNAME="`basename $0`"
RCDIR="`dirname $0`"


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

# ??? should not hard code file suffix
TARGET=$PROJECT_NAME.database/Src/$MODULE_NAME

{
    cat $RCDIR/include/sse.h
	cat << EOF
	#define MOD(a,b) ((a)%(b))
	#define MAX0(a,b) (a>b?a:b)
EOF
	sed -r  -e 's/float (v4sf_[^[]+)/__m128 \1/g' \
			-e 's/float (v4si_[^[]+)/__m128i \1/g' \
            -e 's/v4s[if]_([^,[]+)\[[^]]*\]/\1/g' \
            -e 's/v4s[if]_([^ ,[]+)/\1/g' \
			$TARGET
} > $OUTFILE

cat 1>&2 << EOF
********************************************
substitution done,
output can be compiled using
gcc -march=native -O3 -c $OUTFILE
********************************************
EOF

