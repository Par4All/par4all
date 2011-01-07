from __future__ import with_statement # this is to work with python2.5
from pyps import workspace

with workspace('exception01.c',name='exception01',deleteOnClose=True) as ws:
	try:
		ws.fun.brout.outline(module_name="e",label="i-am-rubber-you-are-glue-everything-bouce-of-me-and-sticks-to-you")
	except RuntimeError , r:
		print "Exception caught:", str(r)

