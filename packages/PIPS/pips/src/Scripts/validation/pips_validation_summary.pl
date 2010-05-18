#! /usr/bin/perl -w
#
# $Id$
#
# further summarize detailed summary

my $failed = 0;
my $timeout = 0;
my $changed = 0;
my $passed = 0;
my $skipped = 0;
my $missing = 0;
my $scripts = 0;
my $sources = 0;
my $orphan = 0;
my $broken = 0;

my %failed = ();
my %timeout = ();
my %changed = ();
my %passed = ();
my %skipped = ();
my %missing = ();
my %scripts = ();
my %sources = ();
my %orphan = ();

my %dir = ();

while (<>)
{
  if (/^timeout: ([-\w]+)/) {
    $timeout++;
    $timeout{$1}++;
    $dir{$1} = 1;
  }
  elsif (/^failed: ([-\w]+)/) {
    $failed++;
    $failed{$1}++;
    $dir{$1} = 1;
  }
  elsif (/^changed: ([-\w]+)/) {
    $changed++;
    $changed{$1}++;
    $dir{$1} = 1;
  }
  elsif (/^passed: ([-\w]+)/) {
    $passed++;
    $passed{$1}++;
    $dir{$1} = 1;
  }
  elsif (/^broken-directory: ([-\w]+)/) {
    $broken++;
    #$broken{$1}++;
  }
  elsif (/^(skipped|orphan|missing|multi-script|multi-source): ([-\w]+)/)
  {
    if ($1 eq 'skipped') {
      $skipped++;
      $skipped{$1}++;
    }
    elsif ($1 eq 'orphan') {
      $orphan++;
      $orphan{$1}++;
    }
    elsif ($1 eq 'missing') {
      $missing++;
      $missing{$1}++;
    }
    elsif ($1 eq 'multi-script') {
      $scripts++;
      $scripts{$1}++;
    }
    elsif ($1 eq 'multi-source') {
      $sources++;
      $sources{$1}++;
    }
    else {
      die "dead end, should not get there";
    }
  }
}

my $count = $timeout + $failed + $changed + $passed;
my $not_passed = $failed + $changed + $timeout;
my $warned = $skipped + $orphan + $missing + $scripts + $sources;

printf
  "total: $count\n" .
  " * passed: $passed\n" .
  " * not passed: $not_passed\n" .
  " - failed: $failed (voluntary and unvoluntary core dumps)\n" .
  " - changed: $changed (modified output)\n" .
  " - timeout: $timeout (time was out)\n" .
  "warnings: $warned\n" .
  " * skipped: $skipped (source without validation scripts)\n" .
  " * missing: $missing (empty result directory)\n" .
  " * multi-script: $scripts (more than one validation script)\n" .
  " * multi-source: $sources (source files for test with different suffixes)\n" .
  " * orphan: $orphan (result avaible without source nor script)\n" .
  "broken directory: $broken (directory without makefile)\n" .
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
  $timeout{$dir} = 0 unless exists $timeout{$dir};
  my $failures = $failed{$dir} + $changed{$dir} + $timeout{$dir};
  my $dircount = $passed{$dir} + $failures;
  my $success_rate = $passed{$dir}*100.0/$dircount;
  printf "%-28s %4d  %4d  %5.1f%%\n", $dir, $dircount, $failures, $success_rate;
}
