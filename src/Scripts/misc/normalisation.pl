#! /usr/local/bin/perl -w
#
# $Id$
#

use Getopt::Long;

$opt_directory = '';
$opt_suffix = 'old';
$opt_help = '';

GetOptions("directory=s", "suffix=s", "help")
    or die $!;

if ($opt_help) {
    print STDERR
	"Usage: normalisation.pl [-d dir] [-s suf] [-h] Logfile\n" .
	"\t-d dir: directory of sources\n" .
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
	    # print "OPENING $current_file\n";
	    
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
	    or die "cannot rename file $current_file ($!)";
	# save to initial file
	open FILE, "> $current_file"
	    or die "cannot open file $current_file for saving ($!)";
	print FILE @fortran;
	close FILE or die $!;
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
