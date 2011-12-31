from __future__ import with_statement # this is to work with python2.5
from pyps import workspace
import sys
with workspace("basics0.c",verbose=False,deleteOnClose=True) as w:
	print >> sys.stderr, "hello"
	for f in w.fun:f.display()
