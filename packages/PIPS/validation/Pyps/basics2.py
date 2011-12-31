from __future__ import with_statement # this is to work with python2.5
#!/usr/bin/env python
from pyps import *
with workspace("basics2.c",deleteOnClose=True) as ws:
	conv=ws.fun.convol
	conv.unfolding()
	lbl=conv.loops(0).loops(0).label
	conv.outline(label=lbl,module_name=(conv.name+"_outlined"))
	ws.all.display()
