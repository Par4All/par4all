#!/usr/bin/perl
use strict;
use warnings;

my $usage = "usage: pipsmakerc2python.pl rc-file ...\n";
if( $#ARGV +1 < 1 ) { die $usage; }

my $rcfile=$ARGV[0];

# read source file file into a string
open INPUT ,$rcfile or die "cannot open $rcfile:$!";
my @lines=<INPUT>; 
my $rc = join "", @lines;
close INPUT;

# pretty_printer
sub print_python_method {
    my $name=$_[0];
    my $doc = $_[1];
    $doc =~s/\n+//m;
    $name =~s/\s/_/;
    print <<EOF

	def $name(self,**props):
		"""$doc"""
		_set_properties(self._update_props("$name",props))
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
my @properties=($rc=~/\\begin{PipsProp}\s*\n+([^ ]+).*?\\end{PipsProp}/gms);
print "\tall_properties=frozenset([";
foreach(@properties) { print "\"$_\","; }
print "\"it's a megablast\"])\n";






