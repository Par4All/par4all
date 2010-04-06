from pyps import *
import terapips
import os,sys

save_dir="tera.out"

def microcode_normalizer(ws,module):
	ws.activate("must_regions")
	ws.activate("transformers_inter_full")
	ws.set_property(array_priv_false_dep_only=False)

	# remove ifs
	module.if_conversion_init()
	module.if_conversion()

	module.array_to_pointer(convert_parameters="1D",flatten_only=True)
	module.loop_normalize(one_increment=True,lower_bound=False)
	module.flatten_code(flatten_code_unroll=False)
	module.partial_eval()
	module.common_subexpression_elimination()
	module.icm()
	module.partial_eval()
	module.suppress_dead_code()
	module.clean_declarations()
	module.simd_atomizer(atomize_reference=True,atomize_lhs=True)
	module.flatten_code(flatten_code_unroll=False)
	module.generate_two_addresses_code()
	module.display()
	module.normalize_microcode()
	for p in ["addi","subi","muli","divi","seti"]:
		module.expression_substitution(pattern=p)
	module.flatten_code(flatten_code_unroll=False)
	module.clean_declarations()
	module.display()

if __name__ == "__main__":
	w = workspace(["alphablending.c", "include/load.c", "include/terasm.c"], cppflags="-I.")
	m = w["alphablending"]
	
	print "tidy the code just in case of"
	m.partial_eval()
	m.display()
	
	#print "tiling"
	#for l in m.loops():
	#	if l.loops():
	#			l.loop_tiling(matrix="128 0,0 8")
	#m.loop_normalize(one_increment=True,skip_index_side_effect=True)
	#m.partial_eval()
	#m.display()
	#m.iterator_detection()
	#m.array_to_pointer(convert_parameters="POINTER",flatten_only=False)
	#m.display(With="PRINT_CODE_PROPER_EFFECTS")
	#m.common_subexpression_elimination(skip_lhs=False)
	#m.simd_atomizer(atomize_reference=True,atomize_lhs=True)
	#m.invariant_code_motion(CONSTANT_PATH_EFFECTS=False)
	#m.icm(CONSTANT_PATH_EFFECTS=False)
	m.display()
	sys.exit()
	
	print "outlining to launcher"
	seed,nb="launcher_",0
	launchers=[]
	for l0 in m.loops():
		for l1 in l0.loops():
			for l2 in l1.loops():
				name=seed+str(nb)
				nb+=1
				m.outline(module_name=name,label=l2.label)
				launchers+=[w[name]]
	m.display()
	for l in launchers:l.display()
	
	print "outlining to microcode"
	microcodes=[]
	for l in launchers:
		theloop=l.loops()[0]
		name=l.name+"_microcode"
		l.outline(module_name=name,label=l.loops()[0].loops()[0].label)
		mc=w[name]
		l.display()
		mc.display()
		microcodes+=[mc]

		print "normalize microcode", mc.name
		microcode_normalizer(w,mc)

	print "saving everything in", save_dir
	w.save(indir=save_dir)
	for m in microcodes:
		path=save_dir+"/"+m.name
		mc.saveas(path+".c")
		terapips.conv(path+".c",file(path+".tera","w"))
		os.remove(path+".c")

