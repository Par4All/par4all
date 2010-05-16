from pyps import *
import terapips
import os,sys

save_dir="tera.out"

def microcode_normalizer(ws,module):
	ws.activate("must_regions")
	ws.activate("transformers_inter_full")
	ws.set_property(array_priv_false_dep_only=False)

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
	module.array_to_pointer(convert_parameters="1D",flatten_only=True)
	module.display()
	module.simd_atomizer(atomize_reference=True,atomize_lhs=True)
	module.display()
	module.generate_two_addresses_code()
	module.display()
	module.normalize_microcode()
	module.display()
	#for p in ["addi","subi","muli","divi","seti"]:
	#	module.expression_substitution(pattern=p)
	#module.flatten_code(flatten_code_unroll=False)
	#module.clean_declarations()
	#module.display()

if __name__ == "__main__":
	w = workspace(["alphablending.c", "include/load.c", "include/terasm.c"], cppflags="-I.")
	m = w["alphablending"]
	
	print "tidy the code just in case of"
	m.partial_eval()
	m.display()
	 
	print "I have to do this early"
	m.terapix_remove_divide()
	m.display()

	
	print "tiling"
	for l in m.loops():
		if l.loops():
				l.loop_tiling(matrix="128 0,0 8")
	print "group constants and isolate"
	kernels=[]
	for l0 in m.loops():
		for l1 in l0.loops():
			for l2 in l1.loops():
				m.group_constants(layout="terapix",statement_label=l2.label,skip_loop_range=True)
				m.display()
				kernels+=[l2.label]
				m.isolate_statement(label=l2.label)
	m.display()
	m.loop_normalize(one_increment=True,skip_index_side_effect=True,lower_bound=0)
	m.display()
	m.partial_eval(linearize=True)
	m.display()
	#m.iterator_detection()
	#m.array_to_pointer(convert_parameters="POINTER",flatten_only=False)
	#m.display(With="PRINT_CODE_PROPER_EFFECTS")
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
		m.outline(module_name=name,label=k,smart_reference_computation=True)
		launchers+=[w[name]]
	m.display()
	for l in launchers:l.display(With='PRINT_CODE_REGIONS')
	
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
