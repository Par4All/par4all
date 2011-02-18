from __future__ import with_statement # this is to work with python2.5
from pyps import module, backendCompiler
import pyps
import terapyps_asm
import pypsutils
from subprocess import Popen, PIPE
import os,sys,shutil,tempfile

def generate_check_ref(self):
	"""Generate a reference run for workspace"""
	(rc,out,err)=self.compile_and_run(backendCompiler(CC="gcc"))
	if rc == 0:
		self.ref=out
	else :
		print err
		exit(1)
pyps.workspace.generate_check_ref=generate_check_ref

def check(self,debug):
	if debug:
		(rc,out,err)=self.compile_and_run(backendCompiler(CC="gcc"))
		if rc == 0:
			if self.ref!=out:
				print "**** check failed *****"
				exit(1)
			else:
				print "**** check ok ******"
		else :
			exit(1)
pyps.workspace.check=check

dma="terapix.c"
assembly="terasm.c"
runtime=["terapix", "terasm"]

class workspace(pyps.workspace):
	"""A Terapix workspace"""
	def __init__(self, *sources, **kwargs):
		"""Add terapix runtime to the workspace"""
		super(workspace,self).__init__(pypsutils.get_runtimefile(dma,"terapyps"),pypsutils.get_runtimefile(assembly,"terapyps"), *sources, **kwargs)

def smart_loop_expansion(m,l,sz,debug,center=False):
	""" smart loop expansion, has a combinaison of loop_expansion_init, statement_insertion and loop_expansion """
	l.loop_expansion_init(loop_expansion_size=sz)
	m.statement_insertion()
	l.loop_expansion(size=sz,center=center)
	m.partial_eval()
	m.invariant_code_motion()
	m.redundant_load_store_elimination()
	if debug:m.display()




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
	"""Generate terapix code for m if it's not part of the runtime """
	if m.cu in runtime:return
	w=m._ws
	w.generate_check_ref()
	# choose the proper analyses and properties
	w.props.constant_path_effects=False
	w.activate(module.must_regions)
	w.activate(module.transformers_inter_full)
	w.activate(module.interprocedural_summary_precondition)
	w.activate(module.preconditions_inter_full)
	w.activate(module.region_chains)
	w.props.semantics_trust_array_declarations=True

	if debug:print "tidy the code just in case of"
	m.partial_eval()
	 
	#print "I have to do this early"
	m.recover_for_loop()
	m.for_loop_to_do_loop()
	if debug:m.display()
	unknown_width="__TERAPYPS_HEIGHT"
	unknown_height="__TERAPYPS_WIDTH"
	m.run(["sed","-e","3 i	unsigned int "
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

	print "group constants and isolate"
	kernels=[]
	for l0 in m.loops():
		for l1 in l0.loops():
			for l2 in l1.loops():
				m.solve_hardware_constraints(label=l2.label,unknown=tiling_vector[0],limit=nbPE,type="NB_PROC")
				m.partial_eval()
				if debug:m.display()
				m.solve_hardware_constraints(label=l2.label,unknown=tiling_vector[1],limit=memoryPE*nbPE,type="VOLUME")
				if debug:m.display()
				m.partial_eval()
				m.forward_substitute()
				m.redundant_load_store_elimination()
				m.clean_declarations()
				m.group_constants(layout="terapix",statement_label=l2.label,skip_loop_range=True)
				if debug:m.display()
				#m.smart_loop_expansion(l2,str(nbPE),debug,center=False)
				m.array_expansion()
				if debug:m.display()
				for k in all_callers(m):
					k.array_expansion()
					if debug:k.display()
				w.check(debug)


	for l0 in m.loops():
		for l1 in l0.loops():
			for l2 in l1.loops():
				kernels+=[l2]
				m.group_constants(layout="terapix",statement_label=l2.label,skip_loop_range=True)
				if debug:m.display()
				m.isolate_statement(label=l2.label)
				if debug:m.display()
	m.loop_normalize(one_increment=True,skip_index_side_effect=True,lower_bound=0)
	m.partial_eval()
	w.check(debug)
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
		m.privatize_module()
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
		name=l.name+"_microcode"
		loop_to_outline=theloop.loops()[0]
		l.outline(module_name=name,label=loop_to_outline.label,smart_reference_computation=True)
		mc=w[name]
		if debug:l.display()
		if debug:mc.display()
		microcodes+=[mc]
	w.check(debug)
	print "refining microcode"
	for m in microcodes:
		m.loop_normalize(one_increment=True,lower_bound=0)
		m.redundant_load_store_elimination()
		m.flatten_code(flatten_code_unroll=False)
		m.linearize_array(use_pointers=True)
		if debug:m.display()
		#m.common_subexpression_elimination()
		#if debug:m.display()
		m.strength_reduction()
		m.forward_substitute()
		m.redundant_load_store_elimination()
		m.split_update_operator()
		#m.expression_substitution("refi")
		#if debug:m.display()
		m.simd_atomizer(atomize_reference=True,atomize_lhs=True)
		#m.generate_two_addresses_code()
		w.check(debug)
		if debug:m.display()
		m.flatten_code(flatten_code_unroll=False)
		m.clean_declarations()
		w.check(debug)
		m.normalize_microcode()
		m.clean_declarations()
		if debug:
			m.display()
			m.callers.display()
	w.check(debug)

	# generate assembly
	for m in microcodes:
		for asm in w.fun:
			if asm.cu == runtime[1]:
				m.expression_substitution(asm.name)
		if debug:m.display()
		terapyps_asm.conv(w.dirname()+m.show("printed_file"),sys.stdout)

module.terapix_code_generation=terapix_code_generation

