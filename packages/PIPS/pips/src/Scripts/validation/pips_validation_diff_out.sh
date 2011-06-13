#!/bin/bash
#
# $Id$
#
# show diffs between test and out files
# usage: $0 path/to/test

test=$1
out=${test%/test}/out
test -f $test || { echo "[$0] missing file: $test"; exit 11; }
test -f $out  || { echo "[$0] missing file: $out"; exit 12; }
diff $test $out
exit $?
