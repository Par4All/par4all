from __future__ import with_statement # this is to work with python2.5
from pyps import workspace

try:
	w=workspace('error.c',name='error',deleteOnClose=True)
	print"you will never see this"
except RuntimeError: print "grrrrr, syntax error"
workspace.delete('error')

