from __future__ import with_statement # this is to work with python2.5
from pyps import *
import terapips
import os,sys

save_dir="tera.out"

def microcode_normalizer(ws,module):

	# remove ifs
	#module.if_conversion_init()
	#module.if_conversion()

	module.loop_normalize(one_increment=True,lower_bound=0,skip_index_side_effect=True)
	module.display()
	module.flatten_code(flatten_code_unroll=False)
	module.display()
	#module.partial_eval()
	#module.display()
	#module.common_subexpression_elimination()
	#module.display()
	#module.icm()
	#module.display()
	#module.partial_eval()
	#module.display()
	#module.suppress_dead_code()
	#module.display()
	#module.clean_declarations()
	#module.display()
	module.linearize_array(use_pointers=True)
	module.display()
	module.normalize_microcode()
	module.display()
	module.simd_atomizer(atomize_reference=True,atomize_lhs=True)
	module.display()
	module.generate_two_addresses_code()
	module.display()
	for p in ["addi","addri","subi","subri","muli","mulri","seti","lshifti","rshifti","psubi","paddi","setpi","prshifti","pseti"]:
		module.expression_substitution(pattern=p)
	module.flatten_code(flatten_code_unroll=False)
	module.clean_declarations()
	module.display()

""" convert a matrix to a string suitable for loop_tiling"""
def vconv(tiling_vector):
	return ",".join(tiling_vector)

""" smart loop expasion, has a combinaison of loop_expansion_init, statement_insertion and loop_expansion """
def smart_loop_expansion(m,l,sz):
	l.loop_expansion_init(loop_expansion_size=sz)
	m.display()
	m.statement_insertion()
	m.display()
	l.loop_expansion(size=sz)
	m.display()

module.smart_loop_expansion=smart_loop_expansion



if __name__ == "__main__":
	with workspace("alphablending2.c", "include/terapix_runtime.c", "include/terasm.c", cppflags="-I.",deleteOnClose=True) as w:
		m = w["alphablending"]
		w.activate(module.must_regions)
		w.activate(module.region_chains)
		w.activate(module.rice_regions_dependence_graph)
		w.activate(module.transformers_inter_full)
		w.props.ARRAY_PRIV_FALSE_DEP_ONLY=False
		w.props.CONSTANT_PATH_EFFECTS=False
		
		print "tidy the code just in case of"
		m.partial_eval()
		 
		print "I have to do this early"
		m.terapix_remove_divide()
		m.display()

		tiling_vector=["128","8"]
		
		print "tiling"
		for l in m.loops():
			if l.loops():
					# this take care of expanding the loop in order to match number of processor constraint
					m.smart_loop_expansion(l,tiling_vector[0])
					# this take care of expanding the loop in order to match memory size constraint
					m.smart_loop_expansion(l.loops()[0],tiling_vector[1])
					# this performs the real tiling
					m.privatize_module()
					l.symbolic_tiling(vector=vconv(tiling_vector))
					m.display()

		print "group constants and isolate"
		kernels=[]
		for l0 in m.loops():
			for l1 in l0.loops():
				for l2 in l1.loops():
					m.group_constants(layout="terapix",statement_label=l2.label,skip_loop_range=True)
					m.display()
					kernels+=[l2]
					m.isolate_statement(label=l2.label)
		m.display()
		m.loop_normalize(one_increment=True,skip_index_side_effect=True,lower_bound=1)
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
			mc=w[name]
			l.display()
			mc.display()
			microcodes+=[mc]

			print "normalize microcode", mc.name
			microcode_normalizer(w,mc)

#	print "saving everything in", save_dir
#	w.save(indir=save_dir)
#	for m in microcodes:
#		path=save_dir+"/"+m.name
#		mc.saveas(path+".c")
#		terapips.conv(path+".c",file(path+".tera","w"))
#		os.remove(path+".c")
#		print "terapix microcode"
#		for line in file(path+".tera"):
#			print line,
#
