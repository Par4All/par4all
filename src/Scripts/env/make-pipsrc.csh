#!/bin/sh

# to derive the C-Shell version of pipsrc.ref

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

  sed '/^test/p;
       /^[ 	]*[A-Za-z_0-9]*=/!d;
       s/\([A-Za-z_0-9]*\)=/setenv \1 /;' pipsrc.sh 

  echo 'rehash' 
} > pipsrc.csh

