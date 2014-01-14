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
#
# 1.Replace unnormalized formal array declarations (assumed-size A(*) and 
# pointer-type A(1)) by their good declarations.
#
# 2.Insert new declarations about common variables created by PIPS
# 
# INTEGER I_PIPS_SUB_ARRAY
# COMMON /PIPS_SUB_ARRAY/ I_PIPS_SUB_ARRAY
#
# 3.Insert new assignments of these variables values before a 
# given statement (the statement ordering is known)
#
# I_PIPS_SUB_ARRAY = actual_array_size
#
# Expected syntax in the instrument file: prefixed tab-separated list 
#
# $COMMON_DECLARATION\tfile\tmodule\tordering
#   code...
# $COMMON_DECLARATION_END
#
# and
# 
# "$ARRAY_DECLARATION\tfile\tmodule\tarray\tnumber_of_dimensions\told_declaration\tnew_declaration"
#
# Command : array_resizing_instrument.pl  < instrument_file

use Getopt::Long;

$opt_help = '';
$opt_suffix = 'old';
GetOptions("help")
    or die $!;

if ($opt_help) {
    print STDERR
	"Usage: array_resizing_instrument.pl < instrument_file\n" .
	"\tinstrument_file: contains new array and common declarations\n" .
	"\tin format of prefixed tab-separated lists \n" .
	"\t: \$ARRAY_DECLARATION file module array ndim old new \n" .
	"\tformat: \$COMMON_DECLARATION file module ordering \n" .
	"\t\tdeclarations or assignments \n" .
	"\t\t\$COMMON_DECLARATION_END \n" .
	"\t-h: this help\n";
    exit;
}
 
# Already seen new declarations SUBROUTINE:ARRAY
%files =();
%seenarray = ();
%seencommon = ();

&debug("Loading declarations and assignments to be inserted ...\n");

while (<>)
{
    if (/^\$ARRAY_DECLARATION/)
    {
	chomp;
	@split = split /\t/;
	shift(@split); 
	($file, $module, $array, $ndim, $old, $new) = @split;
	
	if (exists $seenarray{"$module:$array"})
	{
	    failed("Several declarations for $module:$array\n");
	    next;
	}
	$seenarray{"$module:$array"} = 1;
	
	# only change if old declaration is 1 or * and old != new
	next if $old ne '*' and $old ne '1';
	next if $old eq $new; 
	&debug("Adding array line: $file $module $array $ndim $old $new \n");
	push(@{$files{$file}},"ARRAY:$module:$array:$ndim:$old:$new\n");
    }
    if (/^\$COMMON_DECLARATION/)
    {
	chomp;
	@split = split /\t/;
	shift(@split); 
	($file, $module, $ordering) = @split;
	$code = '';
	while (($line = <>) !~ /^\$COMMON_DECLARATION_END/)
	{
	    $code .= $line;
	}
	if (exists $seencommon{"$module:$ordering:$code"})
	{
	    failed("Several common declarations for $module:$ordering:$code\n");
	    next;
	}
	$seencommon{"$module:$ordering:$code"} = 1;
	&debug("Adding common line: $file $module $ordering \n$code \n");
	push(@{$files{$file}},"COMMON:$module:$ordering:$code\n");
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
	if ($line =~  /^ARRAY:([^:]*):([^:]*):([^:]*):([^:]*):(.*)\n$/s)
	{
	    ($module,$array,$ndim,$old,$new) = ($1,$2,$3,$4,$5);
	    &debug("Considering line: $module $array $ndim $old $new \n");
	    $n = @fortran;
	    $done1 = 0;
	    $insub = 0;
	    for ($i = 0; $i < $n and not $done1; $i++)
	    {
		# Attention when matching a specified module or array
		# => use assertion, if not => confuse between, for example
		# module LIN and LINAVG (turb3d), array AM and AMT (apsi)
		# => add \b after $module and \b before $array 
		$ligne = $fortran[$i];	 
		next if $ligne =~ /^[Cc!\*]/; # skip comments.
		# insure that we're in the right routine
		$insub = 0
		    if $ligne =~ /^[^Cc\*!].*(function|subroutine)/i;
		$insub = 1 
		    if $ligne =~ /^[^Cc\*!].*(function|subroutine)[ \t]*$module\b/i;
		$insub = 1 
		    if $module eq '-';
		next if not $insub;
		# first occurence of ARRAY ( ..  or 
		#  .... ARRAY 
		# &(...) (continuation line)
		# is supposed to be its declaration!
		# 
		# There is a problem with the continuation line here
		# (column 6: character != blank or 0 )
		# bug Array declaration not found for CATHAR 
		# x ,TRUPT
		# &(1:1),DRUPT(I_PIPS_PERMA_DRUPT),EPSCOM(1:2),REC,VOLUME(1:5),REMFT0
		# => OK

		if (($ligne =~ /\b$array *\(/i) ||
		    (($ligne =~ /\b$array */i) && ($fortran[$i+1] =~ /^......\(/)))
		{
		    &debug("Found array $array in module $module at line $i \n");
		    &debug("Line before: $ligne \n");
		    $done1 = 1;
		    if ($ndim>1) 
		    {
			# multi-dimension array
			# if the close parenthese is in the same line
			if ($ligne !~ s/\b($array *\([^()]*),[^,()]*\)/$1,$new\)/i) 
			{
			    # if the close parenthese is in the continuation line
			    if ($fortran[$i+1] =~ /^......\)/i)
			    {
				# if ......) (only the close parenthese)
				$ligne =~ s/\b($array *\([^()]*),[^,()]*/$1,$new/i;
			    }
			    else 
			    {
				# if ........,..) (the last upper bound is completely in the 
				# continuation line, hoping that this bound is not in two lines :-()
				if ($fortran[$i+1] !~ s/(^......[^\)]*)\w+\)/$1$new\)/i)
				{
				    failed("Cannot fix last dimension $ligne \n");
				}
			    }
			}
		    } 
		    else 
		    {
			# one dimension array
			# if the close parenthese is in the same line
			if ($ligne !~ s/\b($array *\()[^()]*\)/$1$new\)/i) 
			{
			    # if the close parenthese is in the continuation line
			    if ($fortran[$i+1] =~ /^......\)/i)
			    {
				# if ......) (only the close parenthese)
				$ligne =~ s/\b($array *\()[^()]*/$1$new/i;
			    }
			    else 
			    {
				# if ........,..) (the last upper bound is completely in the 
				# continuation line, hoping that this bound is not in two lines :-()
				if ($fortran[$i+1] !~ s/(^......[^\)]*)\w+\)/$1$new\)/i)
				{
				    failed("Cannot fix dimension $ligne \n");
				}
			    }
			}

			# bug Found array ZC in module VTRAN2 at line 37 
			# Failed: Cannot fix dimension      
			# &NPAT),AVM(I_PIPS_VTRAN2_AVM),AVP(I_PIPS_VTRAN2_AVP),ZC(1:
			# => Okay
		    }
		    &debug("Line after: $ligne \n");
		    $fortran[$i] = $ligne;
		}
	    }
	    failed("Array declaration not found \n") if not $done1;
	}
	if ($line =~  /^COMMON:([^:]*):([^:]*):(.*)\n$/s)
	{
	    ($module,$ordering,$code) = ($1,$2,$3);
	    &debug("Considering line: $module $ordering \n$code \n");
	    $n = @fortran;
	    $done2 = 0;
	    $insub = 0;
	    for ($i = 0; $i < $n and not $done2; $i++)
	    {
		$ligne = $fortran[$i];	
		# &debug("Print line $ligne \n");
		# insure that we're in the right routine
		#$insub = 0
		 #   if $ligne =~ /^[^Cc\*!].*(program|function|subroutine)/i;
		$insub = 1 
		    if $ligne =~ /^[^Cc\*!].*(program|function|subroutine)[ \t]*$module\b/i;
		#$insub = 1 
		 #   if $module eq '-';
		next if not $insub;
		
		#&debug("In the right routine $module ? $insub \n");
		# find line that matchs current ordering
		if ($fortran[$i] =~ /C \($ordering\)\n/)
		    {
			&debug("Appending code to line $i\n");
			$done2 = 1;
			$fortran[$i] .= "$code";
		    }
	    }
	    failed("Ordering not found \n") if not $done2;   
	}
    }

    # fix line length where necessary...
    # Bug with PERDY_CP1S and VJACO2_CP1S of CATHAR: lines are not well truncated 
    # Before:
    #      &TABVOL(1:9,-45*IV+315),RT(12),RT0(12),QLPAT(NPAT),QVPAT(NPAT),QPAT(NPAT     
    #      &),SPAT(NPAT),HLPAT(1:NPAT,1:1),HVPAT(1:NPAT,1:1),
    # After:
    #      &TABVOL(1:9,-45*IV+315),RT(12),RT0(12),QLPAT(NPAT),QVPAT(NPAT),QPAT
    #      x (NPAT     &),SPAT(NPAT),HLPAT(1:NPAT,1:1),HVPAT(1:NPAT,1:1),

    $n = @fortran;
    for ($i=0; $i<$n; $i++)
    {
	# skip comments
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
	#or die "Cannot open file $new_file for saving ($!)";
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










