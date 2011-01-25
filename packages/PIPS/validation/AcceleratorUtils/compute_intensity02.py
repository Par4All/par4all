from __future__ import with_statement # this is to work with python2.5
from pyps import workspace, module
s='compute_intensity02'
workspace.delete(s)
with workspace(s+'.c',name=s,deleteOnClose=True) as ws:
	ws.activate(module.must_regions)
	ws.activate(module.region_chains)
	ws.props.constant_path_effects=False
	firs=ws.filter(lambda s:s.name.find('fir_') != -1)
	firs.display(activate=module.print_code_complexities)
	firs.computation_intensity(bandwidth=1,frequency=300) # that is we need a factor of 300 between the two
	firs.display()
	firs.privatize_module()
	firs.coarse_grain_parallelization()
	firs.display()



