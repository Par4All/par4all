#!/bin/sh
#
# SCCS Stuff
# $RCSfile: make-pipsrc.sh,v $ ($Date: 1994/03/11 10:59:11 $) version $Revision$, got on %D%, %T%
# %A%
#
# to derive the Shell version of pipsrc.ref
# Adapted from x.csh, Fabien COELHO 11/03/94
#

# do not use environment variables such as $PIPSDIR!
DIR=`pwd`
DIR=`basename $DIR`

if [ "$DIR" != Utilities ]
then
	echo $0 should be executed in its directory: Utilities
	exit 1
fi

cd ..

{
  cat <<-!
	#	   --------------------------------------------------------
	#	   --------------------------------------------------------
	#
	#				    WARNING
	#
	#		  THIS FILE HAS BEEN AUTOMATICALLY GENERATED
	#
	#			       DO NOT MODIFY IT
	#
	#	   --------------------------------------------------------
	#	   --------------------------------------------------------
	!

   cat pipsrc.ref

   echo export `sed -n -e '/^[^#]*=/s/\([^ \t]*\)[ \t]*=.*/\1/p' pipsrc.ref | sort -u | tr '\012' ' '` 

} > pipsrc.sh

#
# that's all
#
