#!/bin/sh

# to derive the C-Shell version of pipsrc.ref

{
  sed '/^[ 	]*[A-Za-z_]*=/!d;s/\([A-Za-z_]*\)=/setenv \1 /' \
	pipsrc.sh 
  echo 'rehash' 
} > pipsrc.csh

