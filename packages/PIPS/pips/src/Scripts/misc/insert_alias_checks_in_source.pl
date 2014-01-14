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

# FC 03/07/2001 for NN
#
# inserts alias check code into sources in a pips database.
# expected syntax on standard input:
# AC: MODULE ORDERING
#   code...
# ACEND

&help('missing database name') unless @ARGV;

$db = $ARGV[0];
shift @ARGV;
%modules = ();

# get all alias checks to add for stdin
&debug("loading ac to be performed\n");

while (<>)
{
    if (/^AC: (.*) (.*)$/)
    {
	$module = $1;
	$ordering = $2;
	$code = '';
	while (($line = <>) !~ /^ACEND/)
	{
	    $code .= $line;
	}

	# module ordering code
	&debug("adding ac for $module $ordering\n$code\n");
	push(@{$modules{$module}}, "$ordering:$code");
    }
}

&debug("for each module, modify its source...\n");

foreach $module (sort keys %modules)
{
    &debug("loading module $module\n");
    # load file
    $source = "$db/$module/$module.pref";
    open FILE, "< $source" or die $!;
    @fortran = <FILE>;
    close FILE or die $!;

    # create an index from orderings to line number
    &debug("creating index for $module\n");
    %index = ();
    for ($i=0; $i<@fortran; $i++)
    {
	if ($fortran[$i] =~ /C +(\(\d+,\d+\))\n/)
	{
	    $index{$1} = $i;
	}
    }

    # modify file
    foreach $modification (@{$modules{$module}})
    {
	if ($modification =~ /^([^:]*):(.*)$/s)
	{
	    ($ordering,$code) = ($1,$2);
	    # find ordering comment in @fortran.
	    # insert code just after.
	    if (not defined($index{$ordering}))
	    {
		die "ordering $ordering not found in $module source $source";
	    }
	    else
	    {
		&debug("appending $ordering to line $index{$ordering}\n");
		$fortran[$index{$ordering}] .= $code;
	    }
	}
	else
	{
	    die "bad $modification";
	}
    }

    # save file
    $result = "$source.alias_checked";
    &debug("saving $module in $result\n");
    open FILE, "> $result" or die $!;
    print FILE @fortran;
    close FILE or die $!;
}

sub help()
{
    print STDERR 
	"insert_alias_checks_in_source.pl db < acfile\n" .
	"   db: pips database name\n" .
	"stdin: alias checks to be added.\n" .
	 " - ", @_, "\n";
    exit 1;
}

sub debug()
{
    print STDERR @_;
}
