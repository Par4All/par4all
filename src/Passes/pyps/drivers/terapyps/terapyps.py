from pyps import module, workspace
import os,sys

save_dir="tera.out"

""" smart loop expasion, has a combinaison of loop_expansion_init, statement_insertion and loop_expansion """
def smart_loop_expansion(m,l,sz):
	l.loop_expansion_init(loop_expansion_size=sz)
	m.display()
	m.statement_insertion()
	m.display()
	l.loop_expansion(size=sz)
	m.display()
	m.invariant_code_motion()
	m.redundant_load_store_elimination()
	m.display()

module.smart_loop_expansion=smart_loop_expansion

def vconv(tiling_vector):
	return ",".join(tiling_vector)

def all_callers(m):
	callers=[]
	for i in m.callers:
		if i not in callers:
			callers.append(i)
			for j in all_callers(i):
				callers.append(j)
	return callers


def terapix_code_generation(m):
	w=m._ws
	w.props.constant_path_effects=False
	w.activate(module.must_regions)
	w.activate(module.transformers_inter_full)
	w.activate(module.interprocedural_summary_precondition)
	w.activate(module.preconditions_inter_full)
	w.activate(module.region_chains)

	print "tidy the code just in case of"
	m.partial_eval()
	 
	#print "I have to do this early"
	m.display()
	m.recover_for_loop()
	m.for_loop_to_do_loop()
	m.display()
	unknown="N"
	m.run(["sed","-e","3 i     int "+unknown+";"])
	m.display()
	tiling_vector=["128",unknown]
	
	print "tiling"
	for l in m.loops():
		if l.loops():
				# this take care of expanding the loop in order to match number of processor constraint
				m.smart_loop_expansion(l,tiling_vector[0])
				# this take care of expanding the loop in order to match memory size constraint
				m.smart_loop_expansion(l.loops()[0],tiling_vector[1])
				l.symbolic_tiling(force=True,vector=vconv(tiling_vector))
				m.display()
				#m.icm()
				m.display()

	print "group constants and isolate"
	kernels=[]
	for l0 in m.loops():
		for l1 in l0.loops():
			for l2 in l1.loops():
				m.display(activate="PRINT_CODE_REGIONS")
				m.solve_hardware_constraints(label=l2.label,unknown=tiling_vector[1],limit=512*128)
				m.display()
				m.run(["psolve"])
				m.display(activate="PRINT_CODE_REGIONS")
				m.partial_eval()
				m.display()
				#m.forward_substitute()
				#m.display()
				m.redundant_load_store_elimination()
	#			m.group_constants(layout="terapix",statement_label=l2.label,skip_loop_range=True)
				m.display(activate="PRINT_CODE_REGIONS")
				for k in all_callers(m):
					k.display(activate="PRINT_CODE_REGIONS")
					k.array_expansion()
				m.display(activate="PRINT_CODE_REGIONS")
				kernels+=[l2]
				m.isolate_statement(label=l2.label)
	m.display()
	m.loop_normalize(one_increment=True,skip_index_side_effect=True,lower_bound=0)
	m.display()
	m.partial_eval()
	m.display()
	#m.iterator_detection()
	#m.array_to_pointer(convert_parameters="POINTER",flatten_only=False)
	#m.display(activate="PRINT_CODE_PROPER_EFFECTS")
	#m.common_subexpression_elimination(skip_lhs=False)
	#m.simd_atomizer(atomize_reference=True,atomize_lhs=True)
	#m.invariant_code_motion(CONSTANT_PATH_EFFECTS=False)
	#m.icm(CONSTANT_PATH_EFFECTS=False)
	#m.display()
	
	print "outlining to launcher"
	seed,nb="launcher_",0
	launchers=[]
	for k in kernels:
		name=seed+str(nb)
		nb+=1
		m.outline(module_name=name,label=k.label,smart_reference_computation=True,loop_bound_as_parameter=k.loops()[0].label)
		launchers+=[w[name]]
	m.display()
	for l in launchers:l.display(activate='PRINT_CODE_REGIONS')
	
	print "outlining to microcode"
	microcodes=[]
	for l in launchers:
		theloop=l.loops()[0]
		name=l.name+"_microcode"
		loop_to_outline=theloop.loops()[0]
		print "label:" , loop_to_outline.label
		l.outline(module_name=name,label=loop_to_outline.label,smart_reference_computation=True)
		m.display()
		mc=w[name]
		l.display()
		mc.display()
		microcodes+=[mc]


module.terapix_code_generation=terapix_code_generation

