from __future__ import with_statement # this is to work with python2.5
from pyps import *
with workspace("callers.c",deleteOnClose=True) as w:
	for c in w.fun.main.callees:print c.name
	w.fun.tenant.callers.display()

