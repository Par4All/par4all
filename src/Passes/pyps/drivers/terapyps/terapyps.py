from pyps import module, workspace
import os,sys

save_dir="tera.out"


def terapix_code_generation(m):
	w=m._ws
	w.activate(module.must_regions)
	w.activate(module.transformers_inter_full)
	w.props.ARRAY_PRIV_FALSE_DEP_ONLY=False
	w.props.CONSTANT_PATH_EFFECTS=False
	
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
	m.display()
	print "group constants and isolate"
	kernels=[]
	for l0 in m.loops():
		for l1 in l0.loops():
			for l2 in l1.loops():
				m.group_constants(layout="terapix",statement_label=l2.label,skip_loop_range=True)
				m.display(activate="print_code_regions")
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
		m.display(activate="PRINT_CODE_REGIONS")
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
		l.partial_eval()
		l.outline(module_name=name,label=loop_to_outline.label,smart_reference_computation=True)
		mc=w[name]
		l.display()
		mc.display()
		microcodes+=[mc]

		print "normalize microcode", mc.name
#		microcode_normalizer(w,mc)

module.terapix_code_generation=terapix_code_generation

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
