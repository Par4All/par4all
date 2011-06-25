from __future__ import with_statement # this is to work with python2.5
from pyps import workspace, module

import os
import openmp

def invoke_function(fu, ws):
	ws.props.memory_effects_only=False
        fu.openmp(verbose=True)
	l = ws.save()
	fi = open(l[0].pop(), 'r')
	text = fi.read()
	os.remove(fi.name)
	return text

if __name__=="__main__":
	workspace.delete('paws_openmp')
	with workspace('paws_openmp.c',name='paws_openmp',deleteOnClose=True) as ws:
		for fu in ws.fun:
			print invoke_function(fu, ws)
