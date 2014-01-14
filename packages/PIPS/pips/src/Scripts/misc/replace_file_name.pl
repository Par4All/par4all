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

# recherche les routines dans les fichiers pour mettre
# a jour un fichier de description des redeclaration.
# voir normalisation.pl
#
# arguments : tous les fichiers sources cibles.
# entree standard : fichier de modification sorti de pips.
# sortie standard : resultat transforme.
#

%modtofile = ();

foreach $file (@ARGV) 
{
    open FILE, "< $file" or die $!;

    while (<FILE>)
    {
	next if /^[Cc\*!]/; # skip comments.
	if (/(function|subroutine|program)[ \t]*(\w*)/i)
	{
	    $mod = $2;
	    $mod =~ tr/a-z/A-Z/;
	    $modtofile{$mod} = $file;
	}
    }

    close FILE or die $!;
}

$, = "\t";

while (<STDIN>)
{
    next if ! /^\$DEC/;
    @l = split /\t/;
    if (exists $modtofile{$l[2]}) {
	$l[1] = $modtofile{$l[2]};
    } else {
	print STDERR "no new file found for $l[2]\n";
    }
    print @l;
}
