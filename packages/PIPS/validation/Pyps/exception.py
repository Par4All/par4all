from __future__ import with_statement # this is to work with python2.5
from pyps import workspace

with workspace('exception.c',name='exception',deleteOnClose=True) as ws:
	try: ws0=workspace(['exception.c'],name='exception')
	except RuntimeError: print "grrrrr, same workspace"

