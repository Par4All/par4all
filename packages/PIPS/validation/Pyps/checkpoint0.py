from pyps import *

with workspace("basics0.c") as w:
	w.fun.foo.inlining(callers="malabar")
	w.fun.megablast.display()
	c0=w.checkpoint()
	w.fun.megablast.display()
	w.fun.foo.inlining(callers="megablast")
	w.fun.megablast.display()
	c1=w.checkpoint()
	w.fun.megablast.display()
	w.restore(c0)
	w.fun.megablast.display()
	w.restore(c1)
	w.fun.megablast.display()
