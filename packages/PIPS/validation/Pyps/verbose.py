from pyps import workspace
import sys
with workspace("basics0.c",verbose=False) as w:
	print >> sys.stderr, "hello"
	for f in w.fun:f.display()
