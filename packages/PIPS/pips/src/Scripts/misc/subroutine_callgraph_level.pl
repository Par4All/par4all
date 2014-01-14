#! /usr/bin/perl -w
#
# $Id$
#
# Copyright 1989-2014 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
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
