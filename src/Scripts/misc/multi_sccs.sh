#! /usr/local/bin/bash
#
# $Id$
#

cmd="sccs $1"
shift 1

#echo $sccs
#exit 0

for f
do
    dir=$(dirname $f)
    name=$(basename $f)
    pushd $dir || exit 1
    #set -x
    eval $cmd $name || exit 2
    popd || exit 3
done

