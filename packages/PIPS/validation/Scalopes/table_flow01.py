from __future__ import with_statement # this is to work with python2.5
from pyps import workspace, module
from re import match
ws="table_flow01"
with workspace(ws+".c","p4a_scmp_stubs.c",name=ws,deleteOnClose=True) as ws:
	ws.props.constant_path_effects=False
	map(ws.activate,[module.must_regions, module.transformers_inter_full, module.interprocedural_summary_precondition, module.preconditions_inter_full])
	ws.fun.main.privatize_module()
	launcher_prefix="P4A_scmp_kernel"
	ws.fun.main.scalopragma(gpu_launcher_prefix=launcher_prefix,
							outline_allow_globals=True)
	ws.fun.main.display()
	kernels=ws.filter(lambda m:match(launcher_prefix,m.name))
	kernels.display()
	try:
		kernels.kernel_load_store(load_function="P4A_scmp_read",
								store_function="P4A_scmp_write",
								allocate_function="P4A_scmp_malloc",
								deallocate_function="P4A_scmp_dealloc")
	except RuntimeError, e:
		print "failed to generate communications with kernel load store:\n", str(e)
	ws.fun.main.display()


