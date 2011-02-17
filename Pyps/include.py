from __future__ import with_statement # this is to work with python2.5
from pyps import workspace

with workspace('include.c',name='include',deleteOnClose=True) as ws:
	pass

