#! /usr/local/bin/perl -wp
#
# $Id$
#
# put a sequence of davinci graphs as one graph.
# use for dumped expressions.
#
BEGIN { undef $/; }
s/\n]\n\n\[\n/\n,\n/sg;
