#! /usr/local/bin/perl -w
#
# $Id$
#
# calcule la profondeur maximale des routines dans un callgraph pips.
# passer en argument un fichier .cg issu du prettyprint pips.
#
# utilisation typique : 
#
#   subroutine_callgraph_level.pl MAIN.cg | sort -n
#

# read all lines.
while (<>)
{
    chomp;

    # extract name and spaces
    $spaces = $_;
    $spaces =~ s/[A-Z0-9_]//g;

    $name = $_;
    $name =~ s/ //g;

    # compute depth...
    $depth = (length($spaces)-1)/4;
    
    # store maximum depth in %depth
    if (! defined($depth{$name})) {
	$depth{$name} = $depth;
    } else {
	# keep the max
	if ($depth{$name}<$depth) {
	    $depth{$name} = $depth;
	}
    }
}

# dump the result.
while (($name,$depth) = each %depth) {
    print "$depth $name\n";
}
