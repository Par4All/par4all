from __future__ import with_statement # this is to work with python2.5
from pyps import workspace
from workspace_gettime import workspace as bworkspace

workspace.delete('benchmark')
with bworkspace('benchmark.c',name='benchmark',deleteOnClose=False) as ws:
	ws.fun.run.benchmark_module()
	ws.fun.run.display()
	a=ws.benchmark(iterations=10,args=["200000"])["run"]



