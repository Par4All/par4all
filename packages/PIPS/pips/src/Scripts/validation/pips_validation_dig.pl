#! /usr/bin/perl -w
#
# $Id$
#
# dig into validation results to identify commits responsible for fails
#
# Usage: $0 SUMMARY files list...
# first file is current status to investigate ideally in reverse date order
#

use strict;

# manage options
my $sort = 1;
my $reverse = 1;

use Getopt::Long;
GetOptions(
  "h|help" => sub { print "$0 SUMMARY_Archive/2011/06/*\n"; exit 0; },
  "s|sort!" => \$sort,
  "r|reverse!" => \$reverse
) or die "option error";

@ARGV = sort @ARGV if $sort;
@ARGV = reverse @ARGV if $reverse;

# the first file scanned is the current status
my $n_fails = 0;

# keep track of svn version info
# project -> version
my %current_version = ();
my %previous_version;
# keep track of the corresponding url
# to retrieve more information if necessary
# project -> full url
my %url = ();

# case status
my %status = ();

# possible url of pips svn server at CRI
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
      my $short = $code;
      $short =~ s/\/trunk$//; # compress output
      # get authors of differing commits on the url *only*, in reverse order
      # there may be several authors if there are several commits
      my $who = `svn log --quiet --revision $pre:$previous $url{$code}`;
      $who =~ s/^\-+\n//sg;
      $who =~ s/r(\d+) \| (\w+) \| .*\n/$1($2),/sg;
      $who =~ s/,$//;
      # this may happen if the validation diff is non-deterministic?
      $who = "$previous?" if $who eq '' and $pre=$previous;
      # this should never happen??
      $who = '???' if $who eq '';
      # format as pips@123(calvin),124(hobbes)
      $vdiff .= "$short\@$who ";
    }
  }
  # this may occur for undeterministic diffs??
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
  if (m,($url/([^@]+))@(\d+),)
  {
    $current_version{$3} = $4;
    $url{$3} = $1; # also keep the detailed url
  }

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
