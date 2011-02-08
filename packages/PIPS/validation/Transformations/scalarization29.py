from __future__ import with_statement # this is to work with python2.5
from pyps import workspace, module
name="scalarization29"
with workspace(name+'.c',name=name,deleteOnClose=True) as ws:
	ws.activate(module.must_regions)
	ws.fun.scalarization29.display()
	ws.fun.scalarization29.scalarization()
	ws.fun.scalarization29.display()
with workspace(name+'.c',name=name,deleteOnClose=True) as ws:
	ws.activate(module.must_regions)
	ws.fun.scalarization29.scalarization(force_out=True)
	ws.fun.scalarization29.display()
