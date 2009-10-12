#!/usr/bin/perl
use strict;
use warnings;

my $usage = "usage: pipsmakerc2python.pl rc-file.tex properties.rc pipsdep.rc ...\n";
if( $#ARGV +1 < 3 ) { die $usage; }

my $texfile=$ARGV[0];


# read propdep file into a string and convert into a map
open INPUT, $ARGV[2];
my @lines=<INPUT>; 
my %pipsdeps = ();
foreach (@lines) {
	/(.*?):\s*(.*)/;
	my $pass=$1;
	my @deps=();
	if(defined $2) {
		@deps=split(/ /,$2);
	}
	for(my $i=0; $i<scalar(@deps);$i++) {
		$deps[$i]=lc($deps[$i]);
	}
	@{$pipsdeps{$pass}} = @deps;
}

close INPUT;

# read properties.rc into a string
my $rcfile=$ARGV[1];
open INPUT ,$rcfile or die "cannot open $rcfile:$!";
@lines=<INPUT>; 
my %pipsprops =();
close INPUT;
foreach(@lines) {
	/\s*(.*?)\s+(.*)/;
	if( defined $1 ) {
		my $propname = uc($1);
		my $default = $2;
		$default=~s/^TRUE$/True/;
		$default=~s/^FALSE$/False/;
		$pipsprops{$propname}=$default;
	}
}
close INPUT;


# read texfile into a string
open INPUT ,$texfile or die "cannot open $texfile:$!";
@lines=<INPUT>; 
my $rc = join "", @lines;
close INPUT;

# method pretty_printer
sub print_python_method {
    my $name=$_[0];
    my $doc = $_[1];
	my $extraparams= "";
	my $extraparamssetter = "";
	if( defined @{$pipsdeps{$name}} and scalar(@{$pipsdeps{$name}})>0) {
		my @props =();
		foreach my $prop (@{$pipsdeps{$name}}) {
			my $short_prop = $prop;
			$short_prop=~s/^$name\_(.*)/\1/;
			my $arg = $short_prop."=".$pipsprops{uc($prop)};
			push @props, $arg;
			$extraparamssetter="\t\tself.ws.set_property(".uc($prop)."=$short_prop)\n$extraparamssetter";
		}
		$extraparams = join("," , @props);
		$extraparams="$extraparams,";
	}
    $doc =~s/(\\begin\{.*?\})|(\\end\{.*?\})|(\\label\{.*?\})//gms;
    $doc =~s/(\\(.*?)\{.*?\})//gms;
    $doc =~s/\\_/_/gms;
	$doc =~s/~/ /gms;
    $doc =~s/\\verb\+(.*?)\+/$1/gms;
	$doc =~s/\\verb\|(.*?)\|/$1/gms;
	$doc =~s/\\verb\/(.*?)\//$1/gms;
	$doc =~s/\\PIPS\{\}/PIPS/gms;
    $name =~s/\s/_/g;
    print <<EOF

	def $name(self,$extraparams **props):
		"""$doc"""
$extraparamssetter
		self.ws._set_property(self.__update_props("$name", props))
		self.apply("$name")

EOF
}
# parse the string for documentation
my @doc_strings=($rc=~/\\begin\{PipsPass\}(.*?)\\end\{PipsPass\}/gms);
foreach(@doc_strings)
{
    /\{([^\}]+)\}[\n]+(.*)/gms;
    print_python_method($1,$2)
}

print "\tall_properties=frozenset([";
foreach(keys %pipsprops) { print "\'$_\',"; }
print "\"it's a megablast\"])\n";






