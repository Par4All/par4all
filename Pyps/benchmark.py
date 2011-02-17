from __future__ import with_statement # this is to work with python2.5
from pyps import workspace, backendCompiler
from workspace_gettime import workspace as bworkspace

workspace.delete('benchmark')
with workspace('benchmark.c',name='benchmark',deleteOnClose=True,parents=[bworkspace]) as ws:
	cep = backendCompiler(CC="gcc",CFLAGS="-O2",args=["200000"])
	(rc,out,err)=ws.compile_and_run(cep)
	ws.fun.run.benchmark_module()
	ws.fun.run.display()
	a=ws.benchmark(cep,10)["run"]



