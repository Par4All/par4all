from pyps import module, workspace
from subprocess import Popen, PIPE
import os,sys

save_dir="tera.out"

""" smart loop expasion, has a combinaison of loop_expansion_init, statement_insertion and loop_expansion """
def smart_loop_expansion(m,l,sz,debug,center=False):
	l.loop_expansion_init(loop_expansion_size=sz)
	if debug:m.display()
	m.statement_insertion()
	if debug:m.display()
	l.loop_expansion(size=sz,center=center)
	if debug:m.display()
	m.partial_eval()
	m.invariant_code_motion()
	m.redundant_load_store_elimination()
	if debug:m.display()

def check_ref(w):
	a_out=w.compile()
	pid=Popen("./"+a_out,stdout=PIPE)
	if pid.wait() == 0:
		return pid.stdout.readlines()
	else :
		exit(1)

def check(w,ref,debug):
	if debug:
		a_out=w.compile()
		pid=Popen("./"+a_out,stdout=PIPE)
		if pid.wait() == 0:
			if ref!=pid.stdout.readlines():
				exit(1)
			else:
				print "**** check ok ******"
		else :
			exit(1)

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


def terapix_code_generation(m,nbPE=128,memoryPE=512,debug=False):
	w=m._ws
	w.props.constant_path_effects=False
	w.activate(module.must_regions)
	w.activate(module.transformers_inter_full)
	w.activate(module.interprocedural_summary_precondition)
	w.activate(module.preconditions_inter_full)
	w.activate(module.region_chains)
	w.props.semantics_trust_array_declarations=True
	ref=check_ref(w)

	if debug:print "tidy the code just in case of"
	m.partial_eval()
	 
	#print "I have to do this early"
	if debug:m.display()
	m.recover_for_loop()
	m.for_loop_to_do_loop()
	if debug:m.display()
	unknown_width="__TERAPYPS_HEIGHT"
	unknown_height="__TERAPYPS_WIDTH"
	m.run(["sed","-e","3 i    unsigned int "
		+ unknown_height + ", " + unknown_width + ";"])
	if debug:m.display()
	tiling_vector=[unknown_height, unknown_width]

	
	print "tiling"
	for l in m.loops():
		if l.loops():
				# this take care of expanding the loop in order to match number of processor constraint
				m.smart_loop_expansion(l,tiling_vector[0],debug)
				# this take care of expanding the loop in order to match memory size constraint
				m.smart_loop_expansion(l.loops()[0],tiling_vector[1],debug)
				l.symbolic_tiling(force=True,vector=vconv(tiling_vector))
				if debug:m.display()
				#m.icm()
				if debug:m.display()

	print "group constants and isolate"
	kernels=[]
	for l0 in m.loops():
		for l1 in l0.loops():
			for l2 in l1.loops():
				if debug:m.display(activate="PRINT_CODE_REGIONS")
				m.solve_hardware_constraints(label=l2.label,unknown=tiling_vector[0],limit=nbPE,type="NB_PROC")
				if debug:m.display()
				m.partial_eval()
				if debug:m.display()
				m.solve_hardware_constraints(label=l2.label,unknown=tiling_vector[1],limit=memoryPE*nbPE,type="VOLUME")
				if debug:m.display()
				m.run(["psolve"])
				if debug:m.display(activate="PRINT_CODE_REGIONS")
				m.partial_eval()
				if debug:m.display()
				m.forward_substitute()
				if debug:m.display()
				m.redundant_load_store_elimination()
				m.clean_declarations()
	#			m.group_constants(layout="terapix",statement_label=l2.label,skip_loop_range=True)
				if debug:m.display(activate="PRINT_CODE_REGIONS")
				m.smart_loop_expansion(l2,str(nbPE),debug,center=False)
				if debug:m.display(activate="PRINT_CODE_REGIONS")
				m.array_expansion()
				if debug:m.display()
				for k in all_callers(m):
					if debug:k.display(activate="PRINT_CODE_REGIONS")
					k.array_expansion()
					if debug:k.display()
				check(w,ref,debug)

	for l0 in m.loops():
		for l1 in l0.loops():
			for l2 in l1.loops():
				if debug:m.display(activate="PRINT_CODE_REGIONS")
				kernels+=[l2]
				m.isolate_statement(label=l2.label)
	if debug:m.display()
	m.loop_normalize(one_increment=True,skip_index_side_effect=True,lower_bound=0)
	if debug:m.display()
	m.partial_eval()
	if debug:m.display()
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
	if debug:m.display()
	if debug:
		for l in launchers:l.display(activate='PRINT_CODE_REGIONS')
	
	print "outlining to microcode"
	microcodes=[]
	for l in launchers:
		theloop=l.loops()[0]
		#l.loop_normalize(one_increment=True,lower_bound=0)
		#l.redundant_load_store_elimination()
		if debug:l.display()
		name=l.name+"_microcode"
		loop_to_outline=theloop.loops()[0]
		print "label:" , loop_to_outline.label
		l.outline(module_name=name,label=loop_to_outline.label,smart_reference_computation=True)
		if debug:m.display()
		mc=w[name]
		if debug:l.display()
		if debug:mc.display()
		microcodes+=[mc]
	check(w,ref,debug)


module.terapix_code_generation=terapix_code_generation

