#! /usr/local/bin/perl -w -n
#
# $Id$
#

BEGIN {
    $current_file = '';
    %seen = ();
}

next if not /^\$DEC/;

chomp;
($file, $module, $array, $ndim, $new, $old) = (split /\t/)[1..6];

# FIX file
$file = $module;
$file =~ tr/A-Z/a-z/;
if ( -f "$file.F" ) {
    $file .= '.F';
} elsif ( -f "$file.f" ) {
    $file .= '.f';
} else {
    print STDERR "ERROR no file found for $module\n";
    $file = '';
}

print STDERR "CONSIDERING $file $module $array $ndim $new $old\n";

if (exists $seen{"$module:$array"})
{
    print STDERR "ERROR several declarations for $module:$array\n";
    next;
}

$seen{"$module:$array"} = 1;

next if $old !~ /:1\)/;
next if $new ne '*' and $old =~ /:$new\)/;

if ($file ne $current_file)
{
    #print STDERR "SWITCHING $current_file to $file";
    &save;

    # tmp get actual file if found...
    $current_file = $file;

    # read file 
    if ($current_file) {
	print "OPENING $current_file\n";

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

for ($i = 0; $i < $n and not $done; $i++)
{
    $ligne = $fortran[$i];

    # insure that we're in the right routine
    $insub = 0 if $ligne =~ /^[^Cc\*!].*(function|subroutine)/i;
    $insub = 1 if $ligne =~ /^[^Cc\*!].*(function|subroutine)[ \t]*$module/i;

    next if not $insub;

    # first occurence of ARRAY( is supposed to be its dimension declaration!
    if ($ligne =~ /[ ,]$array\(/i)
    {
	print STDERR "? $array(in $module): $ligne";
	$done = 1;
	
	if ($ndim>1) 
	{
	    if ($ligne !~ s/($array\([^()]*),[^,()]*\)/$1,$new\)/i) {
		failed('cannot fix last dim', $ligne);
	    }
	} else {
	    if ($ligne !~ s/($array\()[^()]*\)/$1$new\)/i) {
		failed('cannot fix dim', $ligne);
	    }
	}

	print STDERR "! $array(in $module): $ligne";

	$fortran[$i] = $ligne;
    }
}

failed('declaration not found', '') if not $done;

END {
    &save;
}

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

	print "CLOSING $current_file\n";
	print STDERR "SAVING $current_file\n";

	# rename initial file
	rename "$current_file", "$current_file.old"
	    or die "cannot rename file $current_file ($!)";
	# save to initial file
	open FILE, "> $current_file"
	    or die "cannot open file $current_file for saving ($!)";
	print FILE @fortran;
	close FILE or die $!;
    }
}

sub failed
{
    my ($n, $l) = @_;
    print STDERR "ERROR ($n) $file $module $array $ndim $new $old\n$l";
}
