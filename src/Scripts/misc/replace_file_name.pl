#! /usr/local/bin/perl -w
#
# $Id$
#
# recherche les routines dans les fichiers pour mettre
# a jour un fichier de description des redeclaration.
# voir normalisation.pl
#

%modtofile = ();

foreach $file (@ARGV) 
{
    open FILE, "< $file" or die $!;

    while (<FILE>)
    {
	next if /^[Cc\*!]/; # skip comments.
	if (/(function|subroutine|program)[ \t]*(\w*)/i)
	{
	    $mod = $2;
	    $mod =~ tr/a-z/A-Z/;
	    $modtofile{$mod} = $file;
	}
    }

    close FILE or die $!;
}

$, = "\t";

while (<STDIN>)
{
    next if ! /^\$DEC/;
    @l = split /\t/;
    if (exists $modtofile{$l[2]}) {
	$l[1] = $modtofile{$l[2]};
    } else {
	print STDERR "no new file found for $l[2]\n";
    }
    print @l;
}
