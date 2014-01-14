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

#
# Given the instrumenting file, the script does the following things:
# Insert ubs checks before a given statement (the statement ordering is known)
#
# Expected syntax in the instrument file: prefixed tab-separated list 
# $UBS_CHECK: FILE MODULE ORDERING
#   code...
# $UBS_CHECK_END
#
# Command : ubs_checking_instrument.pl < instrument_file

use Getopt::Long;

$opt_help = '';
$opt_suffix = 'old';
GetOptions("help")
    or die $!;

if ($opt_help) {
    print STDERR
	"Usage: ubs_checking_instrument.pl < instrument_file\n" .
	"\tinstrument_file: contains ubs checks\n" .
	"\tin format of prefixed tab-separated lists \n" .
	"\tformat: \$UBS_CHECK file module ordering \n" .
	"\t\t checks\n" .
	"\t\t\$UBS_CHECK_END \n" .
	"\t-h: this help\n";
    exit;
}


# get all ubs checks to add for stdin
&debug("Loading ubs checks to be inserted\n");

%files =();
while (<>)
{
    if (/^\$UBS_CHECK/)
    {
	chomp;
	@split = split /\t/;
	shift(@split); 
	($file, $module, $ordering) = @split;
	$code = '';
	while (($line = <>) !~ /^\$UBS_CHECK_END/)
	{
	    $code .= $line;
	}

	# module ordering code
	#&debug("Adding ubs checks: $file $module $ordering \n$code \n");
	push(@{$files{$file}},"$module:$ordering:$code\n");
    }
}

&debug("For each file, create new instrumented file ... \n");

foreach $file (sort keys %files)
{
    &debug("Loading file $file \n");
    open FILE, "< $file"
	or die "Cannot open file $file for reading ($!)";
    @fortran = <FILE>;
    close FILE or die $!;
    # for each insertion
    foreach $line (@{$files{$file}})
    {
	if ($line =~  /^([^:]*):([^:]*):(.*)\n$/s)
	{
	    ($module,$ordering,$code) = ($1,$2,$3);
	    #&debug("Considering line: $module $ordering \n$code \n");
	    $n = @fortran;
	    $done = 0;
	    $insub = 0;
	    for ($i = 0; $i < $n and not $done; $i++)
	    {
		$ligne = $fortran[$i];
		#if ($i < 150) 
		#{
		 #   &debug("Print line $ligne \n");
		#}

		# insure that we're in the right routine

		# bug : su2cor 
		# 1000  FORMAT(1X,75('+')/' PROGRAM SU2V1COR',' -- MODIFIED HEAT BATH ALGO
		# => conclude : not in right routine :-( 

		#$insub = 0
		#    if $ligne =~ /^[^Cc\*!].*(program|function|subroutine)/i;
		$insub = 1 
		    if $ligne =~ /^[^Cc\*!].*(program|function|subroutine)[ \t]*$module\b/i;
		#$insub = 1 
		#    if $module eq '-';
		next if not $insub;
		#if ($i < 150) 
		#{
		 #   &debug("In right routine \n");
		#}
		
		# find line that matchs current ordering
		if ($fortran[$i] =~ /C \($ordering\)\n/)
		{
		    #&debug("Appending code to line $i\n");
		    $done = 1;
		    $fortran[$i] .= "$code";
		}
	    }
	    failed("Ordering not found \n") if not $done;   
	}
    }

    # fix line length where necessary (length <=72)...
    $n = @fortran;
    for ($i=0; $i<$n; $i++)
    {
	

	#&debug("Fix length  $fortran[$i] \n");

	# skip comments
	# attention, this skips also lines with instrumented code C(2,30) ....code...
	next if $fortran[$i] =~ /^[Cc!\*]/;

	$fortran[$i] =~ s/^(.{72})/$1\n     x /;
	
	while ($fortran[$i] =~ s/(\n[^\n]{72})([^\n])/$1\n     x $2/s) {};
    }

    # For CATHAR, it is better to rename the current file to an old file
    # and replace the current file by the new file
    $old_file = "$file.$opt_suffix";
    if (-f $old_file) {
	$i = 1;
	while (-f "$old_file$i") { $i++; }
	$old_file .= $i;
    }
    # rename initial file
    rename "$file", "$old_file"
	or die "cannot rename file $file as $old_file ($!)";
    
    unlink $file;
    &debug("Save to file $file\n");
    open FILE, "> $file"
	or die "cannot open file $file for saving ($!)"; 
    print FILE @fortran;
    close FILE or die $!;


    # chose new source file name 
    #$new_file = "$file.$opt_suffix";
    #if (-f $new_file) {
#	$i = 1;
#	while (-f "$new_file$i") { $i++; }
#	$new_file .= $i;
#    }
#    &debug("Create new file $new_file\n");
    # save to new file
#    open FILE, "> $new_file"
#	or die "Cannot open file $new_file for saving ($!)";
#    print FILE @fortran;
#    close FILE or die $!;
}

sub failed
{
    print STDERR "Failed: ", @_;
}

sub debug()
{
    print STDERR @_;
}

    











