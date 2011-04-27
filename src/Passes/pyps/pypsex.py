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
	dot_cmd = ["dot","-T"+format, os.path.join(module._ws.dirname,module.show("DOTDG_FILE")),"-o"+of]
	if module._ws.verbose:
		print >> sys.stderr , "Generating image with", dot_cmd
	p = Popen(dot_cmd, stdout = PIPE, stderr = PIPE)
	(out,err) = p.communicate()
	if p.returncode !=0:
		print >> sys.stderr, err
		raise RuntimeError("%s failed with return code %d" % (dot_cmd, ret))
	return (of,out,err)
pyps.module.view_dg=view_dg


def loop_distribution(module,**kwargs):
	module.rice_all_dependence(**kwargs)
	module.internalize_parallel_code(**kwargs)
pyps.module.loop_distribution=loop_distribution

def improve_locality(module,**kwargs):
	module.nest_parallelization(**kwargs)
	module.internalize_parallel_code(**kwargs)
pyps.module.improve_locality=improve_locality

_simdizer_auto_tile=pyps.loop.simdizer_auto_tile
def simdizer_auto_tile(loop,**kwargs):
	loop.module.split_update_operator(**kwargs)
	_simdizer_auto_tile(loop,**kwargs)
pyps.loop.simdizer_auto_tile=simdizer_auto_tile

_simdizer=pyps.module.simdizer
def simdizer(module,**kwargs):
	module._ws.activate(module.must_regions)
	module._ws.activate(module.region_chains)
	module._ws.activate(module.rice_regions_dependence_graph)
	_simdizer(module,**kwargs)
pyps.module.simdizer=simdizer




