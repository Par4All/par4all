#! /usr/bin/perl -w
#
# $Id$
#
# further summarize detailed summary

my $failed = 0;
my $changed = 0;
my $passed = 0;
my $skipped = 0;
my $missing = 0;
my $scripts = 0;
my $sources = 0;
my $broken = 0;

my %failed = ();
my %changed = ();
my %passed = ();
my %skipped = ();
my %missing = ();
my %scripts = ();
my %sources = ();

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
  elsif (/^broken-directory: (\w+)/) {
    $broken++;
    #$broken{$1}++;
  }
  elsif (/^(skipped|missing|multi-script|multi-source): (\w+)/)
  {
    if ($1 eq 'skipped') {
      $skipped++;
      $skipped{$1}++;
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

my $count = $failed + $changed + $passed;
my $not_passed = $failed + $changed;
my $warned = $skipped + $scripts + $sources;

printf
  "total: $count\n" .
  " * passed: $passed\n" .
  " * not passed: $not_passed\n" .
  " - failed: $failed\n" .
  " - changed: $changed\n" .
  "warned: $warned\n" .
  " * skipped: $skipped\n" .
  " * missing: $missing\n" .
  " * multi-script: $scripts\n" .
  " * multi-source: $sources\n" .
  "broken directory: $broken\n" .
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
