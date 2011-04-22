from __future__ import with_statement # this is to work with python2.5
from pyps import workspace
from sac import workspace as sac_workspace
from os import remove
filename="convol3x3"
with workspace(filename+".c", parents=[sac_workspace], driver="sse", deleteOnClose=True) as w:
	m=w[filename]
	m.display()
	m.sac(verbose=True)
	m.display()
	w.goingToRunWith(w.save(rep="save-geekou3"),"save-geekou3")

