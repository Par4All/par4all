# -*- coding: utf-8 -*-
"""
Transformation - specific methods must be placed there.
For instance to enforce a property value, an activate etc before calling a pass
"""

from subprocess import Popen, PIPE
import pyps
import sys, os

def dump_chains_or_dg(module,which="whole_dependence"):
	"""dump textual module's dependence graph or atomic chains, "which" parameter 
	specify which "flavor" you want, for instance "chains" or "effective_dependence" 
	(default is whole_dependence)"""
	generator_name = "print_"+which+"_graph"
	generator = getattr(module,generator_name)
	if generator == None:
		return "Sorry, " + generator_name + " is undefined !"
	generator()	
	filename = os.path.join(module.workspace.dirname,module.show("DG_FILE"))
	read_data = "An error occured"
	with open(filename, 'r') as f:
		read_data = f.read()
	print "// " + which + " for " + module.name
	print read_data
pyps.module.dump_chains_or_dg=dump_chains_or_dg

def dump_chains_or_dg(self, which="whole_dependence"):
    """  """
    for m in self: 
    	m.dump_chains_or_dg(which=which)
pyps.modules.dump_chains_or_dg=dump_chains_or_dg


def view_chains_or_dg(module,format="png"):
	"""view module's dependence graph or atomic chains  in the format specified 
	by ``format'' , not intended to be called direcly, use view_dg or view_chains"""
	of=module.name+"."+format
	dot_cmd = ["dot","-T"+format, os.path.join(module.workspace.dirname,module.show("DOTDG_FILE")),"-o"+of]
	if module.workspace.verbose:
		print >> sys.stderr , "Generating image with", dot_cmd
	p = Popen(dot_cmd, stdout = PIPE, stderr = PIPE)
	(out,err) = p.communicate()
	if p.returncode !=0:
		print >> sys.stderr, err
		raise RuntimeError("%s failed with return code %d" % (dot_cmd, ret))
	return (of,out,err)
pyps.module.view_chains_or_dg=view_chains_or_dg

def view_dg(module,format="png"):
	"""view module's dependence graph in the format specified by ``format''"""
	module.print_dot_dependence_graph()
	return module.view_chains_or_dg(format=format)
pyps.module.view_dg=view_dg

def view_chains(module,format="png"):
	"""view module's dependence graph in the format specified by ``format''"""
	module.print_dot_chains_graph()
	return module.view_chains_or_dg(format=format)
pyps.module.view_chains=view_chains


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


# Unfolding, pyps way ! :-)
def unfold(module,**kwargs):
    while module.callees:
      # We continue to inline every callees while there's at least one 
      # inlining done. We avoid inlining stubs
      one_inlining_done = 0
      for callee in module.callees:
        if not callee.stub_p:
		  callee.inlining(callers=module.name)
		  one_inlining_done+=1
      if one_inlining_done == 0:
      	break;
      
pyps.module.unfold = unfold
def unfold(modules,**kwargs):
    for m in modules:
    	m.unfold()
pyps.modules.unfold = unfold
