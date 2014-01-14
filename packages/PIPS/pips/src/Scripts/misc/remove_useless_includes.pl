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

# HYPOTHESES :
# - une routine par fichier
# - syntaxe simple (pas d'espaces dans les noms de variables...)
# - pas d'include dans les includes...
# - dependances entre include p'tet ok...

use strict;

use Getopt::Long;

my $verbose = 0;
my @includes = ();
my $modify = 0;
my $suffix = '.old';
my $prefix = '!';

GetOptions("i|include=s" => \@includes,
	   "v|verbose+" => \$verbose,
	   "m|modify" => \$modify,
	   "s|suffix=s" => \$suffix,
	   "p|prefix=s" => \$prefix,
	   "h|help" => sub {
	       print STDERR 
		   "$0 [-i dir]* [-vmh] fortran_files\n" .
		   "\t-i dir: include directory search path, default .\n" .
		   "\t-v: verbose\n" .
		   "\t-m: modify by commenting out useless includes\n" .
		   "\t-s suffix: suffix for old file (default '.old')\n" .
		   "\t-p prefix: prefix for commented lines (default '!')\n" .
   	           "\t-h: this help\n";
	       exit 0;
	   })
    or die "invalid option";

# set default include search path
@includes = ('.') unless @includes;

# identificators found
my %identificator = ();

# process a line for identificators
# get_identificators($line_to_process)
# returns a list of identificators found on the line
sub get_identificators($)
{
    my ($line) = @_;

    # comments
    return () if $line =~ /^[!Cc\*]/;

    # continuations 6e colonne
    $line =~ s/^     \S//;
    # string constants
    $line =~ s/^([^\']*)\'[^\']*\'/$1/g;
    # numeric constants
    $line =~ s/\b[\+\-]?\d+\.?\d*([de][\+\-]?\d+)?\b//ig;
    # to lower case
    $line =~ tr/A-Z/a-z/;
    #
    print STDERR "standard line=$line" if $verbose>3;
    my @cuts = split /[^a-z0-9_]+/, $line;
    print STDERR "cut=@cuts\n" if $verbose>4;
    # get all identificators

    return grep (!/^(|if|do|end|endif|enddo|print|while|common
		     |data|dimension|read|subroutine|integer
		     |real|integer|logical|gt|le|lt|ge|eq|ne|\d.*
		     |character|implicit|none|then|else|or|and|call
		     |write|return|continue|format|parameter|double
		     |precision|complex|equivalence|goto)$/x,
		 @cuts);
}

# include files already processed for identificators are not cached
# some id may appear in several files.

# detected dependencies between include files
my %declaration_dependencies = ();

# load_include_file_identificators($include_file_name)
# use options @includes, $verbose
# side effects on %identificators and %declaration_dependencies
# returns nothing
sub load_include_file_identificators($)
{
    my ($filename) = @_;
    my ($src_line, $full_file_name);

    print STDERR "considering include file $filename\n" 
	if $verbose;

    for my $dir (@includes)
    {
	$full_file_name = $dir . "/" . $filename;
	last if -f $full_file_name;
	$full_file_name = '';
    }

    die "include '$filename' not fount in @includes" unless $full_file_name;
    
    open INC, "<$full_file_name" or die $!;
    while ($src_line = <INC>)
    {
	for my $id (get_identificators($src_line))
	{
	    if (exists $identificator{$id} and 
		$identificator{$id} ne $filename)
	    {
		print STDERR 
		    "id $id in '$identificator{$id}' and '$filename'\n"
			if $verbose>1;
		${$declaration_dependencies{$filename}}{$identificator{$id}}=1;
	    }
	    else
	    {
		print STDERR "set id $id in '$filename'\n"
		    if $verbose>1;
		$identificator{$id} = $filename;
	    }
	}
    }
    close INC;
}

for my $file (@ARGV)
{
    print STDERR "considering file $file\n";

    my @file_content = ();
    my %all_includes = ();
    my %useful_includes = ();

    # RESTART ALL FOR EACH FILE
    # sinon des problemes avec les noms de variables reutilises.
    %identificator = ();
    %declaration_dependencies = ();

    open FILE, "<$file" or die $!;

    print STDERR "reading file '$file'\n";
    my $line;
    while ($line = <FILE>)
    {
	push @file_content, $line;

	next if $line =~ /^[Cc!\*]/; # comment

	print STDERR "read line=$line" if $verbose>2;

	if ($line =~ /^      +include *['"]([^\'\"]*)["']/i) 
	{
	    my $name = $1;
	    $all_includes{$name} = 1;
	    load_include_file_identificators($name);
	    next;
	}

	for my $id (get_identificators($line))
	{
	    print STDERR "ID=$id\n" if $verbose>2;
	    if (exists $identificator{$id} and
		exists $all_includes{$identificator{$id}})
	    {
		print STDERR "FOUND in '$identificator{$id}'\n" if $verbose>2;
		$useful_includes{$identificator{$id}} = 1;
	    }
	}
    }
    close FILE;

    my $old_n_useful = 0;
    my $n_useful = keys %useful_includes;
    my $n_total = keys %all_includes;

    print STDERR 
	"before transitive closure on dependencies\n",
	" - useful: ", (join ' ', keys %useful_includes), "\n",
	" - all: ", (join ' ', keys %all_includes), "\n"
	    if $verbose>1;

    # inefficient transitive closure
    while ($old_n_useful!=$n_useful and $n_useful<$n_total)
    {
	for my $f (keys %useful_includes)
	{
	    print STDERR "checking for '$f' dependencies\n"
		if $verbose>2;
	    for my $nf (keys %{$declaration_dependencies{$f}})
	    {
		print STDERR "considering '$f' -> '$nf' dep\n"
		    if $verbose>2;
		if (not exists $useful_includes{$nf} and
		    # if not, it is pretty strange... shoud I warn?
		    exists $all_includes{$nf})
		{
		    $useful_includes{$nf} = 1;
		}
	    }
	}
	$old_n_useful = $n_useful;
	$n_useful = keys %useful_includes;
    }

    print STDERR 
	"after transitive closure on dependencies\n",
	" - useful: ", (join ' ', keys %useful_includes), "\n",
	" - all: ", (join ' ', keys %all_includes), "\n"
	    if $verbose>2;


    print STDOUT 
	"$file summary: keep $n_useful from $n_total\n",
	"useful includes ", (join ' ', sort keys %useful_includes), "\n",
	"all includes ", (join ' ', sort keys %all_includes), "\n";

    # something to be done...
    if ($n_useful<$n_total and $modify)
    {
	# comment out useless includes
	map {
	    if (/^      +include *['"]([^\'\"]*)["']/i) {
		# comment out not used includes
		my $name = $1;
		if (not exists $useful_includes{$name}) {
		    $_ = $prefix . $_;
		}
	    }
	} @file_content;

	# save new file
	rename $file, "$file$suffix";
	open FILE, ">$file" or die $!;
	print FILE @file_content;
	close FILE or die $!;
    }
}
