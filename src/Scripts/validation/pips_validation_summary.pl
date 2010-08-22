#! /usr/bin/perl -w
#
# $Id$
#
# further summarize detailed summary, including differential analysis
#
# Usage: $0 SUMMARY [previous-summary]

use strict;

die "expecting one or two arguments" unless @ARGV <= 2 and @ARGV >= 1;

# all possible validation status
my $status = 'failed|changed|passed|timeout';
# miscellaneous issues
my $others =
    'missing|skipped|multi-script|multi-source|orphan|broken-directory';

sub zeroed()
{
  my $h = {};
  for my $s (split '\|', $status) {
      $$h{$s} = 0;
  }
  return $h;
}

# counts: status -> overall number of cases encountered
my %n = ();
for my $s (split '\|', "$status|$others") {
  $n{$s} = 0;
}

my %d = (); # per-directory: { dir -> { status -> cnt } }
my %new = (); # new state: { dir/case -> status }
my %old = (); # old state: { dir/case -> status }
my %diff = (); # state changes: { dir -> { status/status -> cnt } }
my %changes = (); # changes: { status/status -> cnt }

my $first = $ARGV[0];

while (<>)
{
  if (/^($status|$others): ([-\w]+)(\/[-\w]+)?$/)
  {
    my ($stat, $dir, $case) = ($1, $2, $3);
    $d{$dir} = zeroed() unless exists $d{$dir};
    if ($first eq $ARGV)
    {
      $n{$stat}++;
      $d{$dir}{$stat}++;
      $new{"$dir$case"} = $stat if $stat =~ /^($status)$/;
    }
    else # we are dealing with the "previous" state
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
    }
  }
}

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

my $not_passed = $n{failed} + $n{changed} + $n{timeout};
my $count = $not_passed + $n{passed};
my $warned = $n{skipped} + $n{orphan} + $n{missing} +
    $n{'multi-script'} + $n{'multi-source'};

my $status_changes = '';
for my $sc (sort keys %changes) {
  $status_changes .= " $sc=$changes{$sc}";
}
$status_changes = 'none' unless $status_changes;

printf
  "total: $count\n" .
  " * passed: $n{passed}\n" .
  " * not passed: $not_passed\n" .
  " - failed: $n{failed} (voluntary and unvoluntary core dumps)\n" .
  " - changed: $n{changed} (modified output)\n" .
  " - timeout: $n{timeout} (time was out)\n" .
  # should I hide status changes if it was not computed?
  " * status changes:$status_changes\n" .
  "   .=None P=passed F=failed C=changed T=timeout\n" .
  "warnings: $warned\n" .
  " * skipped: $n{skipped} (source without validation scripts)\n" .
  " * missing: $n{missing} (empty result directory)\n" .
  " * multi-script: $n{'multi-script'} (more than one validation script)\n" .
  " * multi-source: $n{'multi-source'} " .
    "(source files for test with different suffixes)\n" .
  " * orphan: $n{orphan} (result available without source nor script)\n" .
  "broken directory: $n{'broken-directory'} " .
    "(directory without makefile or with makefile errors)\n" .
  "success: %5.1f%%\n" .
  "\n",
  $n{passed}*100.0/$count;

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

# generate summary line for mail subject
print "\n";
if ($n{passed} == $count)
{
  print "SUCCEEDED $count\n";
}
else
{
  print "FAILED $not_passed/$count ",
    "($n{failed}+$n{changed}+$n{timeout})$status_changes\n";
}
