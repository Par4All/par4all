#!/usr/bin/env python

# import everything so that a session looks like tpips one
from pyps import *


# a worspace ID is automagically created ... may be not a good feature
w = workspace(["test.c"])

# you can get module object from the modules table
foo=w["foo"]
bar=w["bar"]
malabar=w["malabar"]
mb=w["megablast"]

# and apply transformation to modules
foo.inlining(callers="bar",PURGE_LABELS=False)

#the good old display, default to PRINTED_FILE, but you can give args
foo.display()
bar.display()
malabar.display()
bar.apply("print_code")

# you can also perform operations on loops
mb.display("loops_file")
for l in mb.loops():
    l.unroll(rate=2)
mb.display()

# access all functions
w.all.partial_eval()
w.all.display()

# recover a list of all labels in the source code ... without pipsing
##
import re # we are going to use regular expressions
label_re = re.compile("^ *(\w+):")
# code gives us a list of line view of module's code
lines=foo.code()
labels=[]
for line in lines:
	m = label_re.match(line)
	if m:
		for label in m.groups(1):
			labels.append(label);

if labels:
	print "found labels:"
	for l in labels: print l
##


# new feature ! save the source code somewhere, so that it can be used after
# the workspace is deleted
w.save(indir="sample")
w.compile(outdir="sample", link=False)

# close *and* delete the workspace
del w
