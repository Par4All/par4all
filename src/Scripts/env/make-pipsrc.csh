#!/bin/sh
#
# $Id$
#
# to derive the C-Shell version of pipsrc.ref
#

sed '/^test/p;
     /^}/p;
     /^#/p;
     /^[ 	]*$/p;
     /^[ 	]*[A-Za-z_0-9]*=/!d;
     s/\([A-Za-z_0-9]*\)=/setenv \1 /;' "$@"

echo 'rehash' 
