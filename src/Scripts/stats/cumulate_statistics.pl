#! /usr/local/bin/perl -wn
#
# $Id$
#
# Accumulate counts reported in a file of format: 
# "<what is counted...>: <number>\n"
#
# Exemple of usage:
#
# cumulate_statistics.pl y.database/*/*.loop_stats
#

($nom, $nombre) = split /: /;
$stats{$nom} += $nombre;

END {
    print "Accumulation of statistics:\n";
    foreach $k (sort keys %stats) {
	print "$k: $stats{$k}\n";
    }
}
