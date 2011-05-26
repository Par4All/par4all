from __future__ import with_statement # this is to work with python2.5
#!/usr/bin/env python
#This script is an adaptation of basics0.py for Pyrops

# import everything so that a session looks like tpips one
from pyrops import pworkspace
import shutil,os,pyrops


# a worspace ID is automagically created ... may be not a good feature
#launcher = pyrops.Launcher() #pyrops.getWorkspaceLauncher(["basics0.c"])
#cp = launcher.getObj()
#w = workspace(["basics0.c"], cpypips = cp)

with pworkspace("basics0.c",deleteOnClose=True) as w:
	# you can get module object from the modules table
	foo=w.fun.foo
	bar=w.fun.bar
	malabar=w.fun.malabar
	mb=w.fun.megablast
	
	# and apply transformation to modules
	foo.inlining(callers="bar",USE_INITIALIZATION_LIST=False)
	
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
	w.all.partial_eval()
	# alternate syntax
	for m in w.fun:m.display()
	
	# recover a list of all labels in the source code ... without pipsing
	##
	import re # we are gonna use regular expression
	label_re = re.compile("^ *(\w+):")
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


pyrops.Launcher.shutdown()
