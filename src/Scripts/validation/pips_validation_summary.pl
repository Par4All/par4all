#! /usr/bin/perl -w
#
# $Id$
#
# further summarize detailed summary, including differential analysis
#
# Usage: $0 [--aggregate] SUMMARY [PREVIOUS_SUMMARY]
#
# Bugs
# - homonymous issue with bug & future

use strict;

# whether to count cases aggregated per upper directory
my $aggregate = 0;

# get options
use Getopt::Long;
GetOptions(
    "aggregate|a!" => \$aggregate,
    "help|h" => sub {
	print "$0 [--aggregate] current [previous]\n";
	exit 0;
    }
) or die "unexpected option ($!)";

# manage arguments
die "expecting one or two arguments" unless @ARGV <= 2 and @ARGV >= 1;
my $summary = $ARGV[0];
my $differential = @ARGV==2;

# all possible validation "status", with distinct first letters
# it may happen with warnings that some cases may be multi state?
my $status =
  'failed|changed|passed|timeout|keptout|bug|later|slow|orphan|noref';

# other miscellaneous issues
my $others =
  'skipped|multi-script|multi-source|nofilter|broken-directory|missing' .
  '|empty-test';

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

my $elapsed_time = 0;

# process input formatted as
while (<>)
{
  # parse: <status>: [some/]dir/case duration
  # "case" may be empty for some status about the whole directory
  # "duration" may be empty for some status and in a transition
  # "some-status: sub/dir/case 123"
  if (/^($status|$others): (([-\w\.]+\/)*?[-\w\.]+)(\/[-\w]+)?( \d+)?$/)
  {
    my ($stat, $dir, $case, $time) = ($1, $2, $4, $5);
    $d{$dir} = zeroed() unless exists $d{$dir};
    $elapsed_time+=$time if defined $time;
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
	# first letter of the state is used to display state changes
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
my $delay = '';
if (defined $start and defined $stop)
{
  $delay = $stop-$start;
  if ($delay<100) { # 0 .. 99 s
    $delay .= 's';
  }
  elsif ($delay < 6000) { # 1.6 .. 99.9 mn
    $delay /= 60.0;
    $delay .= 'mn';
  }
  else { # 1.6 .. NN h
    $delay /= 3600.0;
    $delay .= 'h';
  }
  # keep one digit after dot
  $delay =~ s/(\.\d)\d+/$1/;
}

# count new test cases
for my $c (sort keys %new)
{
  if (not exists $old{$c})
  {
    my $N = uc(substr($new{$c},0,1));
    my $dir=$1 if $c =~ /(.*)\//;
    $changes{".$N"}++;
    $diff{$dir}{".$N"}++;
  }
}

# extract various counts
my $not_passed = $n{failed} + $n{changed} + $n{timeout} + $n{noref};
my $cannot_execute = $n{orphan};
my $count = ${not_passed} + ${cannot_execute} + $n{passed};
my $warned = $n{skipped} + $n{nofilter} +  $n{'multi-script'} + $n{missing} +
    $n{'multi-source'} +  $n{keptout} + $n{bug} + $n{later} + $n{slow} +
    $n{'empty-test'};

# status change summary
my $status_changes = '';
if ($differential) {
  for my $sc (sort keys %changes) {
    $status_changes .= " $sc=$changes{$sc}";
  }
  $status_changes= ' none' unless $status_changes;
}

# print global summary
print
  "number of tests: $count\n" .
  " * passed: $n{passed}\n" .
  " * not passed: $not_passed\n" .
  " - failed: $n{failed} (voluntary and unvoluntary core dumps)\n" .
  " - changed: $n{changed} (modified output)\n" .
  " - timeout: $n{timeout} (time was out)\n" .
  " - noref: $n{noref} (no reference file for comparison)\n" .
  " * cannot execute: $cannot_execute (result without source nor script)\n";

print
  " * status changes:$status_changes\n" .
  "   .=none P=passed F=failed C=changed T=timeout " .
  "K=keptout B=bug L=later S=slow O=orphan N=noref\n"
    if $status_changes;

print
  "number of warnings: $warned\n" .
  " * keptout: $n{keptout} (cannot run test)\n" .
  " * bug: $n{bug} (bugged case)\n" .
  " * later: $n{later} (future test case)\n" .
  " * slow: $n{slow} (cases keptout because they take too much time to run)\n" .
  " * skipped: $n{skipped} (source without validation scripts)\n" .
  " * missing: $n{missing} (empty result directory)\n" .
  " * multi-script: $n{'multi-script'} (more than one validation script)\n" .
  " * multi-source: $n{'multi-source'} " .
    "(source files for test with different suffixes)\n" .
  " * nofilter: $n{nofilter} (tpips2 script without corresponding filter)\n" .
  " * empty-test: $n{'empty-test'} empty 'test' result file\n"
    if $warned;

print
  "broken directory: $n{'broken-directory'} " .
    "(directory without makefile or maybe with makefile errors)\n"
    if $n{'broken-directory'};

my $rate = 100;
$rate = $n{passed}*100.0/$count if $count;
printf "success rate: %5.1f%%\n", $rate;
print "overall elapsed time: $delay\n" if defined $delay and $delay;
print "cumulated elapsed time: $elapsed_time\n";
print "\n";

# possibly aggregate counts on the first directory
if ($aggregate)
{
  my %dc = ();
  for my $dir (sort keys %d)
  {
    my $first = (split /\//, $dir)[0];
    # do directory counts
    for my $s (split '\|', "$status|$others")
    {
      $dc{$first}{$s} += $d{$dir}{$s} if defined $d{$dir}{$s};
    }
    # do differences
    if ($first ne $dir)
    {
      for my $sc (keys %{$diff{$dir}})
      {
	  $diff{$first}{$sc} += $diff{$dir}{$sc};
      }
    }
  }
  %d = %dc;
}

# print detailed per-directory summary
print "directory", " " x 19,
      "cases  bads success (F+C+T+N+O|K+B+L+S) changes...\n";
for my $dir (sort keys %d)
{
  my $failures =
    $d{$dir}{failed} + $d{$dir}{changed} + $d{$dir}{timeout} +
    $d{$dir}{orphan} + $d{$dir}{noref};
  # dircount may be null if all tests are kept out
  my $dircount = $d{$dir}{passed} + $failures;

  my $success_rate = 100;
  $success_rate = $d{$dir}{passed}*100.0/$dircount if $dircount;

  printf "%-28s %4d  %4d  %5.1f%%", $dir, $dircount, $failures, $success_rate;

  # show some details
  if ($success_rate!=100.0 or
      $d{$dir}{keptout} or $d{$dir}{bug} or $d{$dir}{later} or $d{$dir}{slow} or
      (exists $diff{$dir} and $differential))
  {
    printf " (%d+%d+%d+%d+%d|%d+%d+%d+%d)",
      $d{$dir}{failed}, $d{$dir}{changed}, $d{$dir}{timeout},
      $d{$dir}{noref}, $d{$dir}{orphan},
      $d{$dir}{keptout}, $d{$dir}{bug}, $d{$dir}{later}, $d{$dir}{slow};

    if ($differential) {
      for my $change (sort keys %{$diff{$dir}}) {
	print " $change=", $diff{$dir}{$change};
      }
    }
  }

  printf "\n";
}
print "\n";

# generate one summary line for mail subject
if ($n{passed} == $count)
{
  # PASSED?
  print "SUCCESS $count",
    ($n{keptout}+$n{bug}+$n{later}+$n{slow})?
	" ($n{keptout}+$n{bug}+$n{later}+$n{slow})": "",
	"$status_changes $delay\n";
}
else
{
  my $issues = $not_passed+$cannot_execute;
  # maybe the syntax could be: F=nn,C=nn,...
  print "ISSUES $issues/$count " .
        "($n{failed}+$n{changed}+$n{timeout}+$n{noref}+$n{orphan}|" .
	"$n{keptout}+$n{bug}+$n{later}+$n{slow})$status_changes $delay\n";
}
