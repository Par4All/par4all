#!/usr/bin/perl
# this script gather properties dependencies from passes defintion in pipsmake-rc.tex
# the main idea is to split the tex file in section, one per pass, and then look for
# PipsPropRef or PipsProp defintion
# each pass is associated with its referenced or defined properties
# resulting in a $pass: @props\n output
use strict;
use warnings;

my $usage = "usage: $0 pipsmake-rc.tex\n";
if( $#ARGV +1 < 1 ) { die $usage; }

my $rcfile=$ARGV[0];

# read source file file into a string
open INPUT ,$rcfile or die "cannot open $rcfile:$!";
my @lines=<INPUT>; 
my $rc = join "", @lines;
close INPUT;

# parse the string for pass section
my @passes=($rc=~/^\\begin\{PipsPass\}(.*?)\n\\(?:(?:begin\{PipsPass\})|(?:(?:sub)*section)|(?:chapter))/gms);
foreach(@passes) {
	/\{(.*?)\}(.*)/gms;
	my $pass = $1;
	my %seen = ();
	my $content = $2;
	print "$pass:";
	my @props =($content=~/(?:(?:\\PipsPropRef)|(?:\\begin\{PipsProp\}))\{(.*?)\}/gms);
	foreach my $prop (@props) {
		unless($seen{$prop}) {
			$seen{$prop}=1;
			print $prop." ";
		}
	}
	print "\n";
}






