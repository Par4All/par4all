#!/bin/sh

# to derive the C-Shell version of pipsrc.sh
# 09/09/92:	cleaned the sed script [PJ]

# do not use environment variables such as $PIPSDIR!
DIR=`pwd`
DIR=`basename $DIR`

if [ "$DIR" != Utilities ]
then
	echo $0 should be executed in its directory: Utilities
	exit 1
fi

cd ..

cat > pipsrc.csh <<!
#	   --------------------------------------------------------
#	   --------------------------------------------------------
#
#				    WARNING
#
#		  THIS FILE HAS BEEN AUTOMATICALLY GENERATED
#
#			       DO NOT MODIFY IT
#
#		  COMMENTS ARE AVAILABLE IN FILE 'pipsrc.sh'
#
#	   --------------------------------------------------------
#	   --------------------------------------------------------
!

sed '/^[ 	]*[A-Za-z_]*=/!d
     s/\([A-Za-z_]*\)=/setenv \1 /' < pipsrc.sh >> pipsrc.csh

echo 'rehash' >> pipsrc.csh
