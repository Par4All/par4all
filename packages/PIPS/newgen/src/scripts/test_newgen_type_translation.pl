#! /usr/bin/perl -w
#
# $Id$
#
# test type translation...
#

# get initial stuff...
$size=0;
while (<STDIN>)
{
    if (/(.+)\t(\d+)\t.+/) {
	$name[$2] = $1;
	if ($2 > $size) {
	    $size = $2; 
	}
    }
}

# identity.
for ($i=0; $i<$size; $i++)
{
    $tr[$i] = $i;
}

# random changes.
for ($i=7; $i<$size; $i++)
{
    $c = (rand($size-7) % ($size-7)) + 7;

    # exchange...
    $tmp = $tr[$c];
    $tr[$c] = $tr[$i];
    $tr[$i] = $tmp;
}

# output new stuff...
for ($i=0; $i<$size; $i++)
{
    print "$name[$i]\t$tr[$i]\t*\n" if $tr[$i] && $name[$i];
}

# changes files...
foreach $file (@ARGV) 
{
    next if $file =~ /\.old/;

    print STDERR "dealing with '$file'\n";

    rename($file, "$file.old");
    open(IN, "< $file.old");
    open(OUT, "> $file");

    # filter file contents. rather rough indeed...
    # may change some entity names... 
    while (<IN>)
    {
	s/^(\d+) /$tr[$1] /;
	s/([ERT\*])(\d+) /$1$tr[$2] /g;
	print OUT;
    }

    close IN;
    close OUT;
}
