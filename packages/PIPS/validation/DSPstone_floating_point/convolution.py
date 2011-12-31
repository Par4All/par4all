from __future__ import with_statement # this is to work with python2.5
from pyps import workspace
from sac import workspace as sac_workspace
from os import remove
filename="convolution"
with workspace(filename+".c", parents=[sac_workspace], driver="sse",deleteOnClose=True) as w:
	m=w.fun.main
	m.display()
	m.sac()
	m.display()
	a_out=w.simd_compile(rep="d.out")
	remove(a_out)

