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

# Script pour mettre a jour les declarations fortran en fonction
# des bornes trouvees par PIPS. Ajoute eventuellement des includes.

use Getopt::Long;

$opt_directory = '';
$opt_suffix = 'old';
$opt_help = '';
$opt_control = '';

chomp($current_working_directory=`pwd`); # getcwd()?

GetOptions("directory=s", "suffix=s", "control:s", "help")
    or die $!;

if ($opt_help) {
    print STDERR
	"Usage: normalisation.pl [-d dir] [-s suf] [-h] [-c] Logfile\n" .
	"\t-d dir: directory of sources\n" .
	"\t-c [comments]: use sccs\n" .
	"\t-s suffix: suffix for old file (default 'old')\n" .
	"\t-h: this help\n" .
	"format: prefixed tab-separated list\n" .
	"\$DEC file module array ndim new old incs...\n";
    exit;
}

# file being filtered
$current_file = '';

# already seen new declarations SUBROUTINE:ARRAY
#   also SUBROUTINE:COMMON
%seen = ();

while (<>)
{
    # keep logfile lines with this prefix
    next if not /^\$DEC/;

    chomp;

    @split = split /\t/;
    shift(@split); 
    ($file, $module, $array, $ndim, $new, $old, $incs) = @split;
    
    # fix file with source directory if appropriate
    $file = "$opt_directory/$file" if $opt_directory;

    print STDERR "CONSIDERING $file $module $array $ndim $new $old\n";
    
    if (exists $seen{"$module:$array"})
    {
	error("several declarations for $module:$array\n");
	next;
    }
    
    $seen{"$module:$array"} = 1;
    
    # only change if old declaration is :1) or :*)
    next if $old !~ /:[\*1]\)/;
    next if $new ne '*' and $old =~ /:$new\)/;
    
    if ($file ne $current_file)
    {
	#print STDERR "SWITCHING $current_file to $file";
	&save;
	
	# tmp get actual file if found...
	$current_file = $file;
	
	# read file 
	if ($current_file) {

	    print STDERR "OPENED $current_file\n";	

	    open FILE, "< $current_file"
		or die "cannot open file $current_file for reading ($!)";
	    @fortran = <FILE>;
	    close FILE or die $!;  
	} else {
	    @fortran = ();
	}
    }
    
    print STDERR "DOING $file $module $array $ndim $new $old\n";
    
    $n = @fortran;
    $done = 0;
    $insub = 0;
    $nlincs = 0;
    
    for ($i = 0; $i < $n and not $done; $i++)
    {
	$ligne = $fortran[$i];
	
	next if $ligne =~ /^[Cc!\*]/; # skip comments.

	# insure that we're in the right routine
	$insub = 0
	    if $ligne =~ /^[^Cc\*!].*(function|subroutine)/i;
	$insub = 1 
	    if $ligne =~ /^[^Cc\*!].*(function|subroutine)[ \t]*$module/i;
	$insub = 1 
	    if $module eq '-';
	
	next if not $insub;

	# line number of last include
	$nlincs = $i
	    if $ligne =~ /^[^Cc\*!][ \t]*include/i;
	
	# first occurence of ARRAY( 
	#   is supposed to be its dimension declaration!
	if ($ligne =~ /[ ,&]$array *\(/i)
	{
	    print STDERR "? $array(in $module): $ligne";
	    $done = 1;
	    
	    if ($ndim>1) 
	    {
		if ($ligne !~ s/($array *\([^()]*),[^,()]*\)/$1,$new\)/i) {
		    failed('cannot fix last dim', $ligne);
		}
	    } else {
		if ($ligne !~ s/($array *\()[^()]*\)/$1$new\)/i) {
		    failed('cannot fix dim', $ligne);
		}
	    }
	    
	    print STDERR "! $array(in $module): $ligne";
	    
	    $fortran[$i] = $ligne;
	}
    }

    if (defined($incs) and not exists $seen{"$module:$incs"})
    {
	if (! $nlincs) {
	    failed("includes not found for $incs", '');
	} else {
	    # insert include in fortran code.
	    splice @fortran, $nlincs, 0, "      INCLUDE \"$incs\"\n";
	    $seen{"$module:$incs"} = 1;
	}
    }
    
    failed('declaration not found', '') if not $done;
}

# save last file
&save;

sub save
{
    if ($current_file)
    {
	# fix line length where necessary...
	$n = @fortran;
	for ($i=0; $i<$n; $i++)
	{
	    # skip comments
	    next if $fortran[$i] =~ /^[Cc!\*]/;
	    
	    $len = length($fortran[$i]);
	    if ($len > 73) {
		$fortran[$i] =
		    substr($fortran[$i], 0, 72) .
			"\n     # " . 
			    substr($fortran[$i], 72, $len-72);
	    }
	}

	# print "CLOSING $current_file\n";
	print STDERR "SAVING $current_file\n";

	# chose old source file name 
	$old_file = "$current_file.$opt_suffix";
	if (-f $old_file) {
	    $i = 1;
	    while (-f "$old_file.$i") { $i++; }
	    $old_file .= $i;
	}
	# rename initial file
	rename "$current_file", "$old_file"
	    or die "cannot rename file $current_file as $old_file ($!)";

	if ($opt_control) {
	    print STDERR "SCCS EDIT $current_file\n";
	    &sccs("edit", $current_file);
	}

	unlink $current_file;
	open FILE, "> $current_file"
	    or die "cannot open file $current_file for saving ($!)"; 
	print FILE @fortran;
	close FILE or die $!;
	
	if ($opt_control) {
	    &sccs("delget -y'$opt_control'", $current_file);
	}
    }
}

sub error
{
    print STDERR "ERROR ", @_;
    print "ERROR ", @_;
}

sub failed
{
    my ($n, $l) = @_;
    error "($n) $file $module $array $ndim $new $old\n$l";
}

# sccs(what, file)
sub sccs
{
    my ($what,$file) = @_;
    my $dir = '';
    if ($file =~ /^(.*)\/([^\/]*)$/)
    {
	$dir = $1;
	$file = $2;
    }

    my $cmd = "sccs $what $file";

    if ($dir) {
	chdir $dir or die "cannot cd $dir ($!)";
    }

    system($cmd) and die "cannot: $cmd";

    if ($dir) {
	chdir $current_working_directory or 
	    die "cannot cd $current_working_directory ($!)";
    }
}
