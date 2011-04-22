from __future__ import with_statement # this is to work with python2.5
from pyps import workspace
from sac import workspace as sac_workspace
from os import remove
import pypips
filename="ddot_r"
pypips.delete_workspace(filename)

with workspace(filename+".c", "tools.c", parents=[sac_workspace], driver="sse", deleteOnClose=False,name=filename,verbose=True) as w:
	m=w[filename]
	m.display()
	m.sac()
	m.display()
	a_out=w.compile(compiler=w.getsacCompiler(gccCompiler))
	remove(a_out)

