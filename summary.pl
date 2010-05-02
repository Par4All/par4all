#! /usr/bin/perl -w
#
# $Id$
#
# further summarize detailed summary

my $failed = 0;
my $changed = 0;
my $passed = 0;
my $warned = 0;

my %failed = ();
my %changed = ();
my %passed = ();
my %warned = ();

my %dir = ();

while (<>)
{
  if (/^failed: (\w+)/) {
    $failed++;
    $failed{$1}++;
    $dir{$1} = 1;
  }
  elsif (/^changed: (\w+)/) {
    $changed++;
    $changed{$1}++;
    $dir{$1} = 1;
  }
  elsif (/^passed: (\w+)/) {
    $passed++;
    $passed{$1}++;
    $dir{$1} = 1;
  }
  elsif (/^(skipped|multi-script|multi-source): (\w+)/)
  {
    $warned++;
    $warned{$1}++;
  }
}

my $count = $failed + $changed + $passed;

printf
  "total: $count\n" .
  "failed: $failed\n" .
  "changed: $changed\n" .
  "passed: $passed\n" .
  "warned: $warned\n" .
  "success: %5.1f%%\n" .
  "\n",
  $passed*100.0/$count;

print "directory                   cases fails success\n";
for my $dir (sort keys %dir)
{
  # set count if empty
  $failed{$dir} = 0 unless exists $failed{$dir};
  $changed{$dir} = 0 unless exists $changed{$dir};
  $passed{$dir} = 0 unless exists $passed{$dir};
  my $failures = $failed{$dir} + $changed{$dir};
  my $dircount = $passed{$dir} + $failed{$dir} + $changed{$dir};
  my $success_rate = $passed{$dir}*100.0/$dircount;
  printf "%-28s %4d  %4d  %5.1f%%\n", $dir, $dircount, $failures, $success_rate;
}
