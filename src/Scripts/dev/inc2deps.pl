#!/usr/bin/perl -w
#
# $Id$
#
# only keep pips libraries from dependencies "lib1:lib2"

use strict;

my %libs = {};
shift @ARGV;
for my $l (@ARGV) {
  $libs{$l} = 1;
}

while (<STDIN>) {
  my ($l, $d) = split /[:\n]/;
  print if exists $libs{$d} and $l ne $d;
}
