from __future__ import with_statement # this is to work with python2.5
from pyps import workspace, module

with workspace("reduc_atomizer.c") as w:
	m = w.fun.test
	m.display()
	m.reduction_atomization()
	m.display()
