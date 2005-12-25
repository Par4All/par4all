#!/bin/sh
#
# $Id$
#
# to derive the Shell version of pipsrc.ref
#

cat pipsrc.ref

echo export `sed -n '/^[^\[{}#=]*=/s/\([^ \t=]*\)[ \t]*=.*/\1/p' pipsrc.ref | 
             sort -u | 
             tr '\012' ' '` 
