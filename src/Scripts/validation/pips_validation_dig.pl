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
  $vdiff = 'nodiff' unless $vdiff;
  return $vdiff;
}

while (<>)
{
  # hmmm... we're reading the next file
  if (/^on dirs: /)
  {
    $file++;
    %previous_version = %current_version;
  }

  # get svn revision
  $current_version{$2} = $3 if m,$url/([^@]+)@(\d+),;

  # investigate a case
  if (/^(passed|changed|failed|timeout): (\S+)/)
  {
    my ($state, $case) = ($1, $2);

    if (not exists $status{$case} and $file==1 and $state ne 'passed')
    {
      $status{$case} = $state;
      $n_fails++;
    }
    elsif (exists $status{$case} and $file>1)
    {
      if ($status{$case} ne 'passed' and $state eq 'passed')
      {
	# status: not passed -> passed
	print "$case: ", vdiff(), "\n";
	delete $status{$case};
	$n_fails--;
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
