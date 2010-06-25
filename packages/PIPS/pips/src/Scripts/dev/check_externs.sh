#!/bin/bash
cat << EOF
==========================
PIPS extern remover
by serge guelton o(^_-)O
==========================
EOF
if ! test -f "$1/src/Libs/ri-util/ri-util-local.h" ; then
	echo "must be called with the root source dir as first parameter" 1>&2
	exit 1
fi

find $1/src/Libs -name '*.[cylh]' | \
	while read line ; do \
		if  ! `echo $line | grep -q 'local.h$'` ; then
			grep -E -l '^extern\s+\w+\s+\w+\(' $line || grep -E -l '\sextern\s+\w+\s+\w+\(' $line
		fi
	done 1>&2
