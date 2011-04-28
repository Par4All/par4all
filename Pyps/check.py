from __future__ import with_statement # this is to work with python2.5
#!/usr/bin/env python

# import everything so that a session looks like tpips one
from workspace_check import workspace
import shutil,os


# a worspace ID is automagically created ... may be not a good feature
# the with statements ensure correct code cleaning
with workspace("check.c",deleteOnClose=True) as w:

	# you can get module object from the modules table
	foo=w.fun.foo
	bar=w.fun.bar
	malabar=w.fun.malabar
	mb=w["megablast"]
	
	# and apply transformation to modules
	foo.inlining(callers="bar",use_initialization_list=False)
	
	#the good old display, default to PRINTED_FILE, but you can give args
	foo.display()
	bar.display()
	malabar.display()
	bar.print_code()
	
	# you can also preform operations on loops
	mb.display(rc="loops_file")
	for l in mb.loops():
	    l.unroll(rate=2)
	mb.display()
	
	# access all functions
	w.all_functions.partial_eval()
	w.all_functions.display()
	
	# recover a list of all labels in the source code ... without pipsing
	##
	import re # we are gonna use regular expression
	label_re = re.compile(r"\n *(\w+):")
	# code gives us a list of line view of module's code
	line=w.fun.megablast.code
	labels = label_re.findall(line)
	
	if labels:
		print "found labels:"
		for l in labels: print l
	##
	
	
	# new feature ! save the source code somewhere, so that it can be used after
	# the workspace is deleted
	a_out=w.compile()

