#! /usr/local/bin/perl -w
#
# $Id$
#
# one database name as an argument
#

use strict;

#use Getopt::Long;

if (@ARGV != 1) {
    die "Usage: $0 pips-database-name-with-callees-avalaible";
}

my $database = $ARGV[0];

sub dump_hash(\%) {
    my ($hr) = @_;
    while (my ($k, $v) = each %$hr) {
	print STDERR "$k -> @$v\n";
    }
}

use File::Find;

# module name string to file name string
my %file_names = ();

find(
    sub 
     {
	 if (/^CALLEES$/) 
	 {
	     my ($module) = (split '/' , $File::Find::dir)[-1];
	     $file_names{$module} = $File::Find::name;
	 }
	
	 if (-d $File::Find::name)
	 {
	     my ($module) = $_;
	     if (-e "$File::Find::name/$module.f" and not
		 -e "$File::Find::name/CALLEES") {
		 print STDERR "$module.f without callees\n";
	     }
	 }
    }, $database);

#dump_hash %file_names;

# module name string to callees list
my %direct_callees = ();

while (my ($module, $file) = each %file_names)
{
    open CALLEES_FILE, $file or die $!;
    my $callees = <CALLEES_FILE>;
    $callees =~ s/.*\(([^\(\)]*)\).*/$1/;
    $direct_callees{$module} = [ split /[ \"]+/, $callees ];
    close CALLEES_FILE or die $!;
}

#dump_hash %callees;

# $ module -> ( $ callee name -> $one path... )
my %indirect_callees = ();
my %direct_callers = ();

# initialize with direct callees!
while (my ($module, $calleesref) = each %direct_callees)
{
    foreach my $callee (@$calleesref)
    {
	${$indirect_callees{$module}}{$callee} = "$module->$callee";
        ${$direct_callers{$callee}}{$module} = "$callee<-$module";
    }
}

# transitive closure...
my %changed = %direct_callees;
while (%changed)
{
    #print STDERR "new loop\n";
    my %new_changed = ();
    foreach my $modified_module (keys %changed)
    {
	#print STDERR " - modified: $modified_module\n";
	foreach my $caller (keys %{$direct_callers{$modified_module}})
	{
	    #print STDERR "   - caller: $caller\n";
	    # let us update $indirect_callees{$caller}
	    foreach my $callee_callee 
		(keys %{$indirect_callees{$modified_module}})
	    {
		unless (exists ${$indirect_callees{$caller}}{$callee_callee})
	        {
		    #print STDERR "     - adding $caller->$callee_callee\n";
		    ${$indirect_callees{$caller}}{$callee_callee} = 
			"$caller->${$indirect_callees{$modified_module}}{$callee_callee}";
           	    $new_changed{$caller} = 1;
		}
            }
	}
    }
    %changed = %new_changed;
}

#print STDERR "RESULTS:\n";

# module -> count in how many cycles it appears
my %vertex_count = ();

foreach my $module (keys %indirect_callees)
{
    #print STDERR "$module: ",
    #(join ' ', values %{$indirect_callees{$module}}),
    #"\n";

    if (exists ${$indirect_callees{$module}}{$module})
    {
	my $cycle = ${$indirect_callees{$module}}{$module};
	print STDOUT "recursion on $module: $cycle\n";
        foreach my $m (split /->/, $cycle) {
	    $vertex_count{$m}++;
	}       
    }
}

for my $module (keys %vertex_count)
{
    print STDOUT "$module count is $vertex_count{$module}\n";
}
