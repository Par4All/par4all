#! /usr/bin/perl -w
#
# $Id$
#
# dig into validation results to identify commits responsible for fails
#
# Usage: $0 SUMMARY files list...
# first file is current status to investigate ideally in reverse date order
# use -sr options to ensure that it is sorted
#

use strict;

# manage options
my $sort = 0;
my $reverse = 0;

use Getopt::Long;
GetOptions(
  "h|help" => sub { print "$0 -sr SUMMARY_Archive/2011/06/*\n"; exit 0; },
  "s|sort!" => \$sort,
  "r|reverse!" => \$reverse
) or die "option error";

@ARGV = sort @ARGV if $sort;
@ARGV = reverse @ARGV if $reverse;

# the first file scanned is the current status
my $n_fails = 0;

# keep track of svn version info
my %current_version = ();
my %previous_version;

# case status
my %status = ();

my $url = 'https?:\/\/svn\.cri\.(ensmp|mines-paristech)\.fr\/svn';

# current file number
my $file = 0;

# compute svn version difference
# based on global variables %*_version
sub vdiff()
{
  my $vdiff = '';
  for my $code (sort keys %current_version)
  {
    my $current = $current_version{$code};
    my $previous = $previous_version{$code};
    die "wrong file order" unless $current <= $previous;
    if ($current != $previous)
    {
      my $pre = $current+1;
      $code =~ s/\/trunk$//; # compress output
      $vdiff .= "$code\@" .
        ($previous!=$pre? "$pre:$previous ": "$pre ");
    }
  }
  # this may occur for undeterministic diffs
  $vdiff = 'nodiff' unless $vdiff;
  return $vdiff;
}

# process input
while (<>)
{
  # hmmm... I detect that we're reading the next file
  if (/^on dirs: /)
  {
    $file++;
    %previous_version = %current_version;
    # nothing to do if no fails on second round
    last if $file>1 and not $n_fails;
  }

  # get svn revision for anything, based on the svn url
  $current_version{$2} = $3 if m,$url/([^@]+)@(\d+),;

  # investigate a case
  if (/^(passed|changed|failed|timeout): (\S+)/)
  {
    my ($state, $case) = ($1, $2);

    if ($file==1 and $state ne 'passed' and not exists $status{$case})
    {
      # first pass, keep track of non-passing status
      $status{$case} = $state;
      $n_fails++;
    }
    elsif ($file>1 and exists $status{$case})
    {
      # second pass and later, look for previous passed status
      if ($status{$case} ne 'passed' and $state eq 'passed')
      {
	print "$case: ", vdiff(), "\n";
	delete $status{$case};
	$n_fails--;
	# we are done, all issues were found
	last unless $n_fails;
      }
    }
  }
}

# show remaining cases...
for my $case (sort keys %status)
{
  print "$case: ?\n";
}
