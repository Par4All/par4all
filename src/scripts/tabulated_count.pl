#! /usr/bin/perl -wn
#
# $Id$ 
#
# reports the number of tabulated definitions in a file
#

@l = /DT\d+ /g;
$n += @l;

END {
  print "number of tabulated definitions: $n\n";
}
