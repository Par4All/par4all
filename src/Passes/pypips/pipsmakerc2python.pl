#!/usr/bin/perl
use strict;
use warnings;

my $usage = "usage: pipsmakerc2python.pl rc-file.tex properties.rc ...\n";
if( $#ARGV +1 < 2 ) { die $usage; }

my $texfile=$ARGV[0];

# read source file file into a string
open INPUT ,$texfile or die "cannot open $texfile:$!";
my @lines=<INPUT>; 
my $rc = join "", @lines;
close INPUT;

# pretty_printer
sub print_python_method {
    my $name=$_[0];
    my $doc = $_[1];
    $doc =~s/(\\begin{.*?})|(\\end{.*?})|(\\label{.*?})//gms;
    $doc =~s/\\_/_/gms;
	$doc =~s/~/ /gms;
    $doc =~s/\\verb\+(.*?)\+/$1/gms;
	$doc =~s/\\verb\|(.*?)\|/$1/gms;
	$doc =~s/\\verb\/(.*?)\//$1/gms;
	$doc =~s/\\PIPS{}/PIPS/gms;
    $name =~s/\s/_/g;
    print <<EOF

	def $name(self,**props):
		"""$doc"""
		self.ws._set_property(self.__update_props("$name",props))
		self.apply("$name")

EOF
}

# parse the string for documentation
my @doc_strings=($rc=~/%%\@UserManualDocumentation:\s*(.*?)%%!UserManualDocumentation/gms);
foreach(@doc_strings)
{
    /([^\n]+)[\n]+(.*)/gms;
    print_python_method($1,$2)
}

# parse the string for properties
my $rcfile=$ARGV[1];

# read source file file into a string
open INPUT ,$rcfile or die "cannot open $rcfile:$!";
@lines=<INPUT>; 
$rc = join "", @lines;
close INPUT;
my @properties=($rc=~/(^[^\s]+)\s/gms);
print "\tall_properties=frozenset([";
foreach(@properties) { print "\'$_\',"; }
print "\"it's a megablast\"])\n";






