#! /usr/local/bin/perl -wn
#
# $Id$
#

($nom, $nombre) = split /: /;
$stats{$nom} += $nombre;

END {
    print "Accumulation of statistics:\n";
    foreach $k (sort keys %stats) {
	print "$k: $stats{$k}\n";
    }
}
