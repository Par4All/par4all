from __future__ import with_statement # this is to work with python2.5
from pyps import *
import terapips
import os,sys


""" smart loop expasion, has a combinaison of loop_expansion_init, statement_insertion and loop_expansion """
def smart_loop_expansion(m,l,sz):
	l.loop_expansion_init(loop_expansion_size=sz)
	m.display()
	m.statement_insertion()
	m.display()
	l.loop_expansion(size=sz)
	m.display()

module.smart_loop_expansion=smart_loop_expansion

def vconv(tiling_vector):
	return ",".join(tiling_vector)


if __name__ == "__main__":
	w = workspace("average_power.c","include/terapix_runtime.c", cppflags="-I.",deleteOnClose=True)
	w.props.constant_path_effects=False
	w.activate(module.must_regions)
	w.activate(module.transformers_inter_full)
	w.activate(module.interprocedural_summary_precondition)
	w.activate(module.preconditions_inter_full)
	w.activate(module.region_chains)

	m = w["average_power"]
	
	print "tidy the code just in case of"
	m.partial_eval()
	 
	#print "I have to do this early"
	m.display()
	m.recover_for_loop()
	m.for_loop_to_do_loop()
	m.display()
#	w["CplAbs"].inlining()
	m.display()

	tiling_vector=["128","N"]
	
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
				m.solve_hardware_constraints(label=l2.label,unknown=tiling_vector[1],limit=2**14,type="VOLUME")
				m.display()
				m.display(activate="PRINT_CODE_REGIONS")
	#			m.group_constants(layout="terapix",statement_label=l2.label,skip_loop_range=True)
	#			m.display()
				kernels+=[l2]
				m.isolate_statement(label=l2.label, ISOLATE_STATEMENT_EVEN_NON_LOCAL = True)
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
	sys.exit()

		#print "normalize microcode", mc.name
		#microcode_normalizer(w,mc)

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
