#! /usr/bin/perl -w
#
# $Id$
#
# generate combinatorial tests for SPoC scheduling

use strict;
use feature 'switch';

# generate op kind of call for output out, with inputs in0 in1, nth case.
sub genop($$$$$)
{
  my ($op, $out, $in0, $in1, $n) = @_;
  my $call;
  given($op) {
    # input level
    when ('0') { $call = "copy($out, $in0)" }
    when ('1') { $call = "copy($out, $in1)" }
    # spoc level
    when ('2') { $call = "erode_8c($out, $in0, k)" }
    when ('3') { $call = "dilate_8c($out, $in1, k)" }
    # ALU level
    when ('4') { $call = "set_constant($out, $n)" }
    when ('5') { $call = "add_const($out, $in0, $n)" }
    when ('6') { $call = "sub_const($out, $in1, $n)" }
    when ('7') { $call = "mul($out, $in0, $in1)" }
    # threshold level
    when ('8') { $call = "threshold($out, $in0, $n+1, 123, 0)" }
    when ('9') { $call = "threshold($out, $in1, $n+2, 123, 1)" }
    # measurement level, special handling
    when ('a') {
      return
        "  freia_aipo_global_min($in0, x);\n  freia_aipo_copy($out, $in0);\n";
    }
    when ('b') {
      return
	"  freia_aipo_global_max($in1, y);\n  freia_aipo_copy($out, $in1);\n";
    }
    default { die "unexpected op=$op" }
  }
  return "  freia_aipo_$call;\n";
}

sub genfunc($$$$)
{
  my ($left, $right, $temp, $last) = @_;
  my $out0 = ($temp & 1) ? 't0': 'o0';
  my $out1 = ($temp & 2) ? 't1': 'o1';
  my $name = "freia_$left$right$temp$last";
  my $t = 'freia_data2d *';
  open FILE, ">$name.c" or die "cannot open file $name.c: $!";
  print FILE "#include \"freia.h\"\nvoid $name(";
  print FILE ($temp & 1)? "": "$t o0, ";
  print FILE ($temp & 2)? "": "$t o1, ";
  print FILE "$t o2, $t in0, $t in1, int32_t * x, int32_t * y, int32_t * k)\n";
  print FILE "{\n";
  print FILE "  $t t0 = freia_common_create_data(16,128,128);\n" if $temp & 1;
  print FILE "  $t t1 = freia_common_create_data(16,128,128);\n" if $temp & 2;
  print FILE genop($left, $out0, 'in0', 'in1', 0);
  print FILE genop($right, $out1, 'in0', 'in1', 1);
  print FILE genop($last, 'o2', $out0, $out1, 2);
  print FILE "  freia_common_destruct_data(t0);\n" if $temp & 1;
  print FILE "  freia_common_destruct_data(t1);\n" if $temp & 2;
  print FILE "}\n";
  close FILE or die "cannot close file: $!";
}

# 1924 cases
my $count = 0;
# left operation
for my $left ('0' ... '9', 'a', 'b')
{
  # right operation, skip 1369b because of same/different symmetry
  for my $right ('0', '2', '4', '5', '7', '8', 'a')
  {
    # skip left/right symmetry
    next if $right lt $left;
    # whether left/right computations are temporaries or outputs
    for my $temp (0 .. 3)
    {
      # last operation
      for my $last ('0' .. '9', 'a', 'b')
      {
	# skip source symmetry
	next if $left eq $right and $last =~ /^[1369b]$/;
	print STDERR "generating $left$right$temp$last\n";
	genfunc($left, $right, $temp, $last);
	$count++;
      }
    }
  }
}

print STDERR "$count functions generated\n";
