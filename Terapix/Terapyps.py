#!/usr/bin/env python
from __future__ import with_statement # this is to work with python2.5
from pyps import *
from sys import exit
import terapips

verbose=2
save_dir="teraout"

def microcode_normalizer(ws,module):
	ws.activate("must_regions")
	ws.activate("transformers_inter_full")
	ws.set_property(array_priv_false_dep_only=False)

	# remove ifs
	module.if_conversion_init()
	module.if_conversion()
	

	module.array_to_pointer(convert_parameters=True,flatten_only=True)
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
	for p in ["addf","subf","mulf","divf","paddf","addi","subi","muli","divi","seti","setf"]:
		module.expression_substitution(pattern=p)
	module.flatten_code(flatten_code_unroll=False)
	module.clean_declarations()
	module.display()




def terapyps(kernel_module,loop_label,*input_sources):
	if verbose == 1 : print "Creating workspace from", list(input_sources)
	# first create a workspace from the input sources
	tws = workspace(*input_sources ,"include/load.c", "include/terasm.c", verboseon=(verbose==2),cppflags="-I.", deleteOnClose=True)
	# then select the given module
	km = tws[kernel_module]
	km.display()
	# first normalize the input source code
	km.partial_eval()
	if verbose == 2: km.display()
	km.unfolding()
	if verbose == 2: km.display()
	km.scalarization()
	if verbose == 2: km.display()
	km.privatize_module()
	#km.array_to_pointer(convert_parameters=False,flatten_only=True)
	#km.common_subexpression_elimination(skip_lhs=False)
	#km.simd_atomizer(atomize_reference=True,atomize_lhs=True)
	#km.invariant_code_motion(partial_distribution=False)
	if verbose == 2: km.display()

	# tile the loop flaged 
	km.loop_normalize(one_increment=True,skip_index_side_effect=True,lower_bound=0)
	km.partial_eval(always_simplify=True)
	km.display()
	kernel=km.loops(loop_label)
	kernel.loop_expansion(size=126) #,trick_region="// remove this one later")
	if verbose == 2: kernel.display()
	kernel.loop_tiling("126 0,0 5")
	if verbose == 2: km.display(With="PRINT_CODE_REGIONS")
	
	launchers_loop = []
	for l0 in kernel.loops():
		for l1 in l0.loops():
			launchers_loop += [l1]
			km.isolate_statement(label=l1.label)
			if verbose == 2: l1.display()
#
	# be cleaner
	km.loop_normalize(one_increment=True,skip_index_side_effect=True)
	km.partial_eval()
	if verbose == 2: km.display(With="PRINT_CODE_REGIONS")
	if verbose == 2: km.display(With="PRINT_CODE_PRECONDITIONS")
	exit()
#	km.suppress_trivial_test()
#	if verbose == 2: km.display()
	km.privatize_module()
	if verbose == 2: km.display()

	# outline lauchers
	launchers,seed,nb=[],"launcher",0
	for launcher_loop in launchers_loop:
			name=seed+str(nb)
			km.outline(label=launcher_loop.label,module_name=name)
			if verbose == 2: km.display()
			if verbose == 2: tws[name].display()
			launchers+=[tws[name]]
			nb+=1
#
#	km.kernel_load_store(load_function="memload",
#			store_function="memstore",
#			allocate_function="memalloc",
#			deallocate_function="memfree")
#	if verbose == 2: km.display()
#
	# // lauchers
	for kl in launchers:
		kl.common_subexpression_elimination()
		kl.display()
		kl.privatize_module()
		# perform loop expansion on inner loop
		for l1 in kl.loops():
			l1.loop_expansion(size=128,offset=1)
			if verbose == 2: l1.display()
		# from launcher to microcode
		for l0 in kl.loops():
			for l1 in l0.loops():
				name=kl.name+"_microcode"
				kl.privatize_module()
				kl.outline(label=l1.label, module_name=name)
				if verbose == 2: kl.display()
				if verbose == 2: tws[name].display()

#		# last steps on microcode
#		microcode=tws[kl.name+"_microcode"]
#		if verbose == 2: microcode.display()
#		microcode_normalizer(tws,microcode)
#
#	# save workspace
#	tws.save(indir=save_dir)
#	for ml in launchers:
#		name=ml.name+"_microcode"
#		newname=save_dir+"/"+name
#		tws[name].saveas(newname+".c")
#		terapips.conv(newname+".c",file(newname+".tera","w"))
#		os.remove(newname+".c")

	print "resulting files printed out in" , save_dir
if __name__ == "__main__":
	terapyps("convol", "here" , "teraconv.c")



