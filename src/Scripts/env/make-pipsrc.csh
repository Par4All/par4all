#!/bin/sh
#
# $Id$
#
# to derive the C-Shell version of pipsrc.ref
#

{
  cat ${PIPS_ROOT}/Include/auto-number.h

  sed '/^test/p;/^}/p;/^#/p;/^[ 	]*$/p;
       /^[ 	]*[A-Za-z_0-9]*=/!d;
       s/\([A-Za-z_0-9]*\)=/setenv \1 /;' pipsrc.sh 

  echo 'rehash' 
} > pipsrc.csh
