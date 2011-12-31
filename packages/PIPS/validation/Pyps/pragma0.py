from __future__ import with_statement # this is to work with python2.5
from pyps import workspace, module
import p3

with workspace("pragma0.c",deleteOnClose=True) as w:
	w.fun.transfo.display()
	w.all.p3()
	w.fun.transfo.display()

