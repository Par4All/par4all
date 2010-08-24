#! /usr/bin/perl -w
#
# $Id$
#
# further summarize detailed summary, including differential analysis
#
# Usage: $0 SUMMARY [previous-summary]

use strict;

# manage arguments
die "expecting one or two arguments" unless @ARGV <= 2 and @ARGV >= 1;
my $summary = $ARGV[0];

# all possible validation status
my $status = 'failed|changed|passed|timeout';

# other miscellaneous issues
my $others =
    'missing|skipped|multi-script|multi-source|orphan|broken-directory';

# return ref to zero count status hash
sub zeroed()
{
  my $h = {};
  for my $s (split '\|', $status) {
    $$h{$s} = 0;
  }
  return $h;
}

# case status
my %new = (); # new state: { dir/case -> status }
my %old = (); # old state: { dir/case -> status }

# record state changes
my %diff = (); # per dir state changes: { dir -> { status/status -> cnt } }
my %changes = (); # state changes: { status/status -> cnt }

# status counts: status -> overall number of cases encountered
my %n = ();
for my $s (split '\|', "$status|$others") {
  $n{$s} = 0;
}

# per directory status counts
my %d = (); # per-directory: { dir -> { status -> cnt } }

my ($start, $stop);

# process input formatted as
# <status>: dir/case
while (<>)
{
  if (/^($status|$others): ([-\w]+)(\/[-\w]+)?$/)
  {
    my ($stat, $dir, $case) = ($1, $2, $3);
    $d{$dir} = zeroed() unless exists $d{$dir};
    if ($summary eq $ARGV) # this is the current summary
    {
      $n{$stat}++;
      $d{$dir}{$stat}++;
      $new{"$dir$case"} = $stat if $stat =~ /^($status)$/;
    }
    else # we are dealing with the "previous" summary
    {
      # extract differential information...
      if ($stat =~ /^($status)$/)
      {
	# record previous state
	$old{"$dir$case"} = $stat;
	my $O = uc(substr($stat,0,1));
	if (exists $new{"$dir$case"})
	{
	  my $news = $new{"$dir$case"};
	  if ($news ne $stat) # just record state changes
	  {
	    # old and new state as one letter F C P T
	    my $N = uc(substr($news,0,1));
	    # record status changes
	    $changes{"$O$N"}++;
	    $diff{$dir}{"$O$N"}++;
	  }
	}
	else # case does not exist anymore
	{
	  $changes{"$O."}++;
	  $diff{$dir}{"$O."}++;
	}
      }
      # else the line is not about the validation status, we ignore it
    }
  }

  # extract time information
  if (not defined $start or not defined $stop)
  {
    $start = $1 if /^start date: .*\[(\d+)\]$/;
    $stop = $1 if /^end date: .*\[(\d+)\]$/;
  }
}

# compute elapsed time and adjust units
my $delay = $stop-$start;
if ($delay<100) {
  $delay .= 's';
}
elsif ($delay < 6000) {
  $delay /= 60.0;
  $delay .= 'mn';
}
else {
  $delay /= 3600.0;
  $delay .= 'h';
}
$delay =~ s/(\.\d)\d+/$1/;

# count new test cases
for my $c (sort keys %new)
{
  if (not exists $old{$c})
  {
    my $N = uc(substr($new{$c},0,1));
    my $dir = (split /\//, $c)[0];
    $changes{".$N"}++;
    $diff{$dir}{".$N"}++;
  }
}

# extract various counts
my $not_passed = $n{failed} + $n{changed} + $n{timeout};
my $count = $not_passed + $n{passed};
my $warned = $n{skipped} + $n{orphan} + $n{missing} +
    $n{'multi-script'} + $n{'multi-source'};

# status change summary
my $status_changes = '';
for my $sc (sort keys %changes) {
  $status_changes .= " $sc=$changes{$sc}";
}

# print global summary
printf
  "number of tests: $count\n" .
  " * passed: $n{passed}\n" .
  " * not passed: $not_passed\n" .
  " - failed: $n{failed} (voluntary and unvoluntary core dumps)\n" .
  " - changed: $n{changed} (modified output)\n" .
  " - timeout: $n{timeout} (time was out)\n" .
  # should I hide status changes altogether if it was not computed?
  " * status changes:" . ($status_changes? $status_changes: " none") . "\n" .
  "   .=None P=passed F=failed C=changed T=timeout\n" .
  "number of warnings: $warned\n" .
  " * skipped: $n{skipped} (source without validation scripts)\n" .
  " * missing: $n{missing} (empty result directory)\n" .
  " * multi-script: $n{'multi-script'} (more than one validation script)\n" .
  " * multi-source: $n{'multi-source'} " .
    "(source files for test with different suffixes)\n" .
  " * orphan: $n{orphan} (result available without source nor script)\n" .
  "broken directory: $n{'broken-directory'} " .
    "(directory without makefile or maybe with makefile errors)\n" .
  "success rate: %5.1f%%\n" .
  "elapsed time: $delay\n" .
  "\n",
  $n{passed}*100.0/$count;

# print detailed per-directory summary
print "directory                   cases  bads success (F+C+T) changes...\n";
for my $dir (sort keys %d)
{
  my $failures = $d{$dir}{failed} + $d{$dir}{changed} + $d{$dir}{timeout};
  my $dircount = $d{$dir}{passed} + $failures;

  printf "%-28s skipped?\n", $dir and next unless $dircount;

  my $success_rate = $d{$dir}{passed}*100.0/$dircount;

  printf "%-28s %4d  %4d  %5.1f%%", $dir, $dircount, $failures, $success_rate;

  if ($success_rate!=100.0 or exists $diff{$dir})
  {
    printf " (%d+%d+%d)",
      $d{$dir}{failed}, $d{$dir}{changed}, $d{$dir}{timeout};
    for my $change (sort keys %{$diff{$dir}})
    {
      print " $change=", $diff{$dir}{$change};
    }
  }

  printf "\n";
}
print "\n";

# generate one summary line for mail subject
if ($n{passed} == $count)
{
  print "SUCCEEDED $count$status_changes $delay\n";
}
else
{
  print "FAILED $not_passed/$count ",
    "($n{failed}+$n{changed}+$n{timeout})$status_changes $delay\n";
}
