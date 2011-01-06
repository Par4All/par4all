from __future__ import with_statement # this is to work with python2.5
from pyps import workspace
from os import remove
import pypips

filename="partialeval03"
pypips.delete_workspace(filename)
with workspace(filename+".c", parents=[], deleteOnClose=False,name=filename) as w:
	m=w['main']
	m.partial_eval()
	m.display()

