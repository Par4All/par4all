#! /usr/bin/perl -w
#
# $Id$
#
# sh> $0 foo.database FOO > foo.dot
# sh> dot -Tpdf -Grankdir=LR foo.dot > foo.pdf
#
# Copyright 1989-2014 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#

use strict;

# manage options
my $debug = 0;
use Getopt::Long;
GetOptions("debug+" => \$debug)
  or die "invalid option ($!)";

# get arguments
die "usage: $0 database-dir module-name\n" unless @ARGV == 2;

my $dbdir = shift;
die "no database directory $dbdir" unless -d $dbdir;

my $module = shift;
die "no such module $module" unless -d "$dbdir/$module";

# callees that are already processed
my %dones = ();

# recursive show
sub show_callees($)
{
  my ($module) = @_;
  print STDERR "considering module $module\n" if $debug;

  $dones{$module} = 1;

  print "  \"$module\" [shape=box];\n";

  # get module's callees
  my @callees = ();
  # ??? I should rather parse CALLEES...
  my $cgfile = "$dbdir/$module/$module.cg";
  open CG, "<$cgfile" or die "cannot open $cgfile";
  while (<CG>)
  {
    # only keep first level...
    next unless /^\s{5}\S/;
    s/\s+//g;
    push @callees, $_;
  }
  close CG;

  # generate arcs, and recursion
  for my $callee (@callees)
  {
    print "  \"$module\" -> \"$callee\";\n";
    show_callees($callee) unless $dones{$callee};
  }
}

# work
print "digraph \"${module}_cg\" {\n";
show_callees($module);
print "}\n";
