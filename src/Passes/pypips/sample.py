#!/usr/bin/env python

# import everything so that a session looks like tpips one
from pyps import *


# a worspace ID is automagically created ... may be not a good feature
create("test.c")

# you can get module object from the modules table
foo=modules["foo"]
bar=modules["bar"]

# and apply transformation to modules
foo.apply("inlining")

# code gives us a list of line view of modue's code
foo.code()

#the good old display, default to PRINTED_FILE, but you can give args
foo.display()
bar.display()
bar.apply("print_code")

# recover a list of all labels in the source code ... without pipsing
##
import re # we are gonne use regular expression
label_re = re.compile("^ *(\w+):")
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
foo.save()

#funny way to save the whole file
module("test","test.c").save("test2.c")

# close *and* delete the workspace
close()
