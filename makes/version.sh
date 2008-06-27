#! /bin/sh
# $Id$

for dir
do
    if type svnversion > /dev/null
    then
	if [ -d $dir/.svn ]
	then
	    echo $(svn info $dir | sed -n -e '2s/.*: //p')@$(svnversion $dir)
	else
	    echo 'unknown@unknown'
	fi
    else
	echo 'unknown@unknown'
    fi
done
