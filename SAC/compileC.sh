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
	typedef __m128 v4sf;
	typedef __m128d v2df;
	typedef __m128i v4si;
	typedef __m128i v8hi;
	typedef float a4sf[4] __attribute__((aligned(16)));
	typedef double a2df[4] __attribute__((aligned(16)));
	typedef int a4si[4] __attribute__((aligned(16)));
	typedef short a8hi[4] __attribute__((aligned(16)));
EOF
    cat $TARGET
} > $OUTFILE

cat 1>&2 << EOF
********************************************
substitution done,
output can be compiled using
gcc -march=native -O3 -c $OUTFILE
********************************************
EOF

