#! /usr/local/bin/perl -w
#
# $Id$
#
# HYPOTHESES :
#
# - une routine par fichier
# - syntaxe simple (pas d'espaces dans les noms de variables...)
# - pas d'include dans les includes...
# - dependances entre include p'tet ok...
#

use Getopt::Long;
$verbose = 0;

GetOptions("i|include=s" => \@includes,
	   "v|verbose+" => \$verbose,
	   "m|modify" => \$modify,
	   "h|help" => sub {
	       print STDERR 
		   "$0 [-i dir] [-vmh] fortran_files\n" .
		   "\t-i dir: include directory search path\n" .
		   "\t-v: verbose\n" .
		   "\t-m: modify by commenting out useless includes\n" .
   	           "\t-h: this help\n";
	       exit 0;
	   })
    or die "invalid option";

# set default include search path
@includes = ('.') unless @includes;

my %identificator = ();

# process a line for identificators
sub get_identificators($)
{
    my ($line) = @_;
    # comment
    return () if $line =~ /^[!Cc*]/;
    # continuations
    $line =~ s/^......//;
    # string constants
    $line =~ s/^([^\']*)\'[^\']*\'/$1/g;
    # numeric constants
    $line =~ s/\b[\+\-]?\d+\.?\d*([de][\+\-]?\d+)?\b//ig;
    return grep (!/^(|if|do|end|endif|enddo|print|while||common
		     |data|dimension|print|read|subroutine|integer
		     |real|integer|logical|gt|le|lt|ge|eq|ne|\d.*
		     |character|implicit|none|then|else|or|and|call
		     |write|return|continue|format|parameter|double
		     |precision|complex|equivalence)$/ix,
		 (split /[^A-Za-z0-9_]+/, $line));
}

# include files already processed for identificators are not cached
# some id may appear in several files.

# detected dependencies between include files
my %declaration_dependencies = ();

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
		# tell anyway, as it might be a 
		print "id $id in $identificator{$id} and $filename\n";
		${$declaration_dependencies{$identificator{$id}}}{$filename}=1;
	    }
	    else
	    {
		print STDERR "set id $id in $filename\n" if $verbose>1;
		$identificator{$id} = $filename;
	    }
	}
    }
    close INC;
}

for $file (@ARGV)
{
    print "considering file $file\n";

    my @file_content = ();
    my %all_includes = ();
    my %useful_includes = ();

    # RESTART ALL FOR EACH FILE
    # sinon des problemes avec les noms de variables reutilises.
    %identificator = ();
    #%loaded_include_files = ();
    %declaration_dependencies = ();

    open FILE, "<$file" or die $!;

    print "reading file $file\n";
    while ($line = <FILE>)
    {

	push @file_content, $line;

	next if $line =~ /^[Cc!\*]/; # comment

	print "line=$line" if $verbose>2;

	if ($line =~ /^      +include *['"]([^\'\"]*)["']/i) 
	{
	    my $name = $1;
	    $all_includes{$name} = 1;
	    load_include_file_identificators($name);
	    next;
	}

	for $id (get_identificators($line))
	{
	    print "ID=$id\n" if $verbose>2;
	    if (exists $identificator{$id} and
		exists $all_includes{$identificator{$id}})
	    {
		print "FOUND in '$identificator{$id}'\n" if $verbose>2;
		$useful_includes{$identificator{$id}} = 1;
	    }
	}
    }
    close FILE;

    my $old_n_useful = 0;
    my $n_useful = keys %useful_includes;
    my $n_total = keys %all_includes;

    # inefficient transitive closure
    while ($old_n_useful!=$n_useful and $n_useful<$n_total)
    {
	for my $f (keys %useful_includes)
	{
	    for my $nf (keys %{$declaration_dependencies{$f}})
	    {
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

    print STDOUT 
	"$file summary: keep $n_useful from $n_total\n",
	"useful includes ", (join ' ', sort keys %useful_includes), "\n",
	"all includes ", (join ' ', sort keys %all_includes), "\n";

    if ($n_useful<$n_total and $modify)
    {
	map {
	    if (/^      +include *['"]([^\'\"]*)["']/i) {
		# comment out not used includes
		my $name = $1;
		if (not exists $useful_includes{$name}) {
		    $_ = '!' . $_;
		}
	    }
	} @file_content;

	# save file
	rename $file, "$file.old";
	open FILE, ">$file" or die $!;
	print FILE @file_content;
	close FILE or die $!;
    }
}
