# -*- coding: utf-8 -*-
"""
Transformation - specific methods must be placed there.
For instance to enforce a property value, an activate etc before calling a pass
"""

from subprocess import Popen, PIPE
import pyps
import sys

def view_dg(module,format="png"):
	"""view module's dependence graph in the format specified by ``format''"""
	module.print_dot_dependence_graph()
	of=module.name+"."+format
	dot_cmd = ["dot","-T"+format, module._ws.dirname()+module.show("DOTDG_FILE"),"-o"+of]
	if module._ws.verbose:
		print >> sys.stderr , "Generating image with", dot_cmd
	p = Popen(dot_cmd, stdout = PIPE, stderr = PIPE)
	(out,err) = p.communicate()
	if p.returncode !=0:
		print >> sys.stderr, err
		raise RuntimeError("%s failed with return code %d" % (dot_cmd, ret))
	return (of,out,err)

pyps.module.view_dg=view_dg





