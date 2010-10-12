#!/usr/bin/env python
#This script is an adaptation of basics0.py for Pyrops

# import everything so that a session looks like tpips one
from pyrops import pworkspace
import shutil,os,pyrops


# a worspace ID is automagically created ... may be not a good feature
#launcher = pyrops.Launcher() #pyrops.getWorkspaceLauncher(["basics0.c"])
#cp = launcher.getObj()
#w = workspace(["basics0.c"], cpypips = cp)

with pworkspace("basics0.c") as w:
	# you can get module object from the modules table
	foo=w.fun.foo
	bar=w.fun.bar
	malabar=w.fun.malabar
	mb=w.fun.megablast
	
	# and apply transformation to modules
	foo.inlining(callers="bar",PURGE_LABELS=False)
	
	#the good old display, default to PRINTED_FILE, but you can give args
	foo.display()
	bar.display()
	malabar.display()
	bar.print_code()
	
	# you can also preform operations on loops
	mb.display("loops_file")
	for l in mb.loops():
	    l.unroll(rate=2)
	mb.display()
	
	# access all functions
	w.all.partial_eval()
	# alternate syntax
	for m in w.fun:m.display()
	
	# recover a list of all labels in the source code ... without pipsing
	##
	import re # we are gonne use regular expression
	label_re = re.compile("^ *(\w+):")
	# code gives us a list of line view of modue's code
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
	w.compile(rep="basics0", link=False)

# tidy ..
shutil.rmtree("basics0")
os.remove("basics0.o")

pyrops.Launcher.shutdown()
