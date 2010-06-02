# get everythin we nedd from pyps
from pyps import *


# helper to recover entity_user_name
def module_user_name(m):
	index=m.name.rfind('!')
	return m.name if index==-1 else m.name[index+1:]

# the list of transformation to apply
# partial eval is too costly to be applied on a whole function,
# so we outline the outer loops and run transformation sequence on them
def run_sac(ws,module):
	for l in module.loops():
		mlabel=l.label
		mname=module_user_name(module)+mlabel
		module.outline(label=mlabel,module_name=mname)
		new_module=ws[mname]
		new_module.if_conversion_init()
		new_module.if_conversion()
		new_module.if_conversion_compact()
		new_module.simd_atomizer()
		new_module.simdizer_auto_unroll(simple_calculation=False,minimize_unroll=False)
		new_module.partial_eval(always_simplify=True)
		new_module.simd_remove_reductions()
		new_module.single_assignment()
		new_module.common_subexpression_elimination(skip_added_constant=True,skip_lhs=False)
		new_module.simdizer()
		new_module.simd_loop_const_elim()
		new_module.clean_declarations()

if __name__ == "__main__":

	# create a sac workspace with appropriate extra includes and activates
	ws = workspace(["linpack.c","include/clock.c","include/SIMD.c"],
			name='lp',
			activates=["RICE_SEMANTICS_DEPENDENCE_GRAPH", "MUST_REGIONS", "PRECONDITIONS_INTER_FULL", "TRANSFORMERS_INTER_FULL"]
			)
	# some more restructuring properties
	ws.set_property(
			UNSPAGHETTIFY_WHILE_RECOVER=True,
			FOR_TO_WHILE_LOOP_IN_CONTROLIZER=False,
			RICEDG_STATISTICS_ALL_ARRAYS=True)
	
	# sac global properties
	ws.set_property(SIMD_FORTRAN_MEM_ORGANISATION=False,SAC_SIMD_REGISTER_WIDTH=128)
	
	# first try to recover as much do loops as possible
	# sac if-conversion fails in presence of while loops :'(
	ws.all.restructure_control()
	ws.all.recover_for_loop()
	
	# run the transformation sequence on each kernel as listed here
	for m in ["ddot", "dscal", "daxpy"]:
		run_sac(ws,ws["linpack!"+m+"_r"])
		run_sac(ws,ws["linpack!"+m+"_ur"])
	
	# display the result here
	for m in ["ddot", "dscal", "daxpy"]:
		map(lambda name:ws["linpack!"+name].display(), [m+"_r",m+"_ur"])

	# save result, otherwise everything will be deleted
	ws.save(indir="linpack_sac")
	# clean & tidy
	ws.close()
	
