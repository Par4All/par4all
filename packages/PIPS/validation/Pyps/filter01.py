from __future__ import with_statement # this is to work with python2.5
from pyps import workspace

with workspace('filter01.c',name='filter01',deleteOnClose=True) as w:
	print map(lambda m:m.name,w.filter(lambda x:True))
