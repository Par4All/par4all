#!/bin/sh
#
# $Id$
#
# to derive the Shell version of pipsrc.ref
#


{
   cat ${PIPS_ROOT}/Include/auto-number.h pipsrc.ref
   echo export `sed -n '/^[^\[{}#=]*=/s/\([^ \t=]*\)[ \t]*=.*/\1/p' pipsrc.ref | sort -u | tr '\012' ' '` 

} > pipsrc.sh

#
# that's all
#
