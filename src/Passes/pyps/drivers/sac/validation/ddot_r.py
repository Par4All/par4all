from __future__ import with_statement # this is to work with python2.5
import pyps
from sac import workspace
from os import remove
import pypips
filename="ddot_r"
pypips.delete_workspace(filename)

with workspace(filename+".c", "tools.c", driver="sse", deleteOnClose=False,name=filename,verbose=True) as w:
	m=w[filename]
	m.display()
	m.sac()
	m.display()
	a_out=w.compile(compiler=w.get_sacCompiler(pyps.gccCompiler)())
	#a_out2=w.compile()
