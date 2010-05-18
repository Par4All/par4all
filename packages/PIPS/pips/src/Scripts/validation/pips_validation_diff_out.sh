#!/bin/bash
#
# $Id$
#
# show diffs between test and out files
# usage: $0 path/to/test

test=$1
out=${test%/test}/out
diff $test $out
