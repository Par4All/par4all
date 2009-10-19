#!/usr/bin/perl
use strict;
use warnings;

my $usage = "usage: pipsmakerc2python.pl rc-file.tex properties.rc pipsdep.rc ... [-loop|-module|-modules]\n";
if( $#ARGV +1 < 4 ) { die $usage; }

my $texfile=$ARGV[0];
my $generator=$ARGV[$#ARGV];


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
	my $has_loop_label=0;
	if( defined @{$pipsdeps{$name}} and scalar(@{$pipsdeps{$name}})>0) {
		my @props =();
		foreach my $prop (@{$pipsdeps{$name}}) {
			my $short_prop = $prop;
			$short_prop=~s/^$name\_(.*)/$1/;
			my $arg = $short_prop."=".$pipsprops{uc($prop)};
			if( $prop eq "loop_label" ) {
				$has_loop_label=1;
				$extraparamssetter="\t\tself.ws.set_property(".uc($prop)."=self.label)\n$extraparamssetter";
			}
			else {
				push @props, $arg;
				$extraparamssetter="\t\tself.ws.set_property(".uc($prop)."=$short_prop)\n$extraparamssetter";
			}
		}
		if( scalar(@props) > 0 ) {
			$extraparams = join("," , @props);
			$extraparams="$extraparams,";
		}
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
	my $self = "self";
	if(($has_loop_label == 1)  and ($generator eq "-loop") ) {
		$self="self.module";
	}
	if( (($has_loop_label == 1)  and ($generator eq "-loop") ) or ( ($has_loop_label == 0) and (not $generator eq "-loop") ) ) {
    	print <<EOF

	def $name(self,$extraparams **props):
		"""$doc"""
$extraparamssetter
EOF
	;

		if( not $generator eq "-modules" ) {
    		print <<EOF
		$self.ws._set_property($self._update_props("$name", props))
		$self.apply("$name")

EOF
			;
		}
		else {
    		print <<EOF
		for m in self.modules:
			m.ws._set_property(m._update_props("$name", props))
			m.apply("$name")

EOF
			;
		}
	}
}
# parse the string for documentation
my @doc_strings=($rc=~/\\begin\{PipsPass\}(.*?)\\end\{PipsPass\}/gms);
foreach(@doc_strings)
{
    /\{([^\}]+)\}[\n]+(.*)/gms;
    print_python_method($1,$2)
}

if( $generator eq '-module' ) {
	print "\tall_properties=frozenset([";
	foreach(keys %pipsprops) { print "\'$_\',"; }
	print "\"it's a megablast\"])\n";
}






