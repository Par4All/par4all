from __future__ import with_statement # this is to work with python2.5
from pyps import workspace
from os import remove
import pypips

filename="pragma"
pypips.delete_workspace(filename)
with workspace(filename+".c", parents=[], driver="sse", name=filename) as w:
	m=w[filename]
	m.suppress_dead_code()
	m.display()

