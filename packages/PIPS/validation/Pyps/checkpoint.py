from __future__ import with_statement # this is to work with python2.5
from pyps import *

with workspace("basics0.c",deleteOnClose=True) as w:
	w.fun.foo.inlining(callers="malabar")
	c0=w.checkpoint()
	w.fun.foo.inlining(callers="bar")
	w.fun.megablast.display()
	w.restore(c0)
	w.fun.megablast.display()
	w.fun.foo.inlining(callers="megablast")
	w.fun.megablast.display()
