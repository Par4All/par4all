from pyps import workspace, module
import p3

with workspace("pragma0.c") as w:
	w.fun.transfo.display()
	w.all.p3()
	w.fun.transfo.display()

