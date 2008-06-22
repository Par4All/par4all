#! /bin/sh
# $Id$

dir=$1

if type svnversion > /dev/null
then
    if [ -d $dir/.svn ]
    then
	svnversion $dir
    else
	echo 'unknown'
    fi
else
    echo 'unknown'
fi
