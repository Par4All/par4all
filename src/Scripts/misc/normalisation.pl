#! /usr/local/bin/perl -w -n
#
# $Id$
#

next if not /^\$DEC/;

($prefix,$file, $module, $ndim, $new, $old) = split /\t/;

print "file=$file, module=$module\n";
