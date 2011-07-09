from __future__ import with_statement # this is to work with python2.5
from pyps import workspace, module
from os import system

with workspace("silber.c","include/adds.c",verbose=False,deleteOnClose=True) as w:
	# print out all functions
	w.all_functions.display()
	#a_out=w.compile()
	#system("./"+a_out+" include/input.pgm include/mapfile.amp /dev/null")

	# print all function with both callers and callees
	for fun in w.all:
		if fun.callers and fun.callees:
			fun.display()
	
	# get some help
	#help(module.unfolding)
	# apply what we have just learnt
	w.fun.transfo.unfolding()
	# select a transformation
	t = w.fun.transfo
	t.display()
	t.flatten_code()
	t.display()
	# this will stop when force_loop_fusion fails
	try:
		while len(t.loops()) > 0:
			l=t.loops(0)
			l.force_loop_fusion()
			t.suppress_dead_code()
			l.display()
	except:pass
	# if everything is ok ...
	if len(t.loops()) == 1:
		outline_module="inner"
		t.forward_substitute(optimistic_clean=True)
		t.display()
		t.common_subexpression_elimination(skip_lhs=False)
		t.suppress_dead_code()
		t.display()
		# greate, saturated arithmeteic !
		#help(module.expression_substitution)
		t.expression_substitution(pattern=w.fun.adds.name)
		t.display()
		t.privatize_module()
		t.coarse_grain_parallelization()
		# great ! OMP pragmas
		t.display()
		#o=w.compile()
		#w.run(o, args=["include/input.pgm","include/mapfile.amp", "/dev/null"])

		w.props.constant_path_effects=False
		lbl=t.loops(0).loops(0).label
		t.outline(module_name=outline_module,label=lbl)
		t.display()
		w[outline_module].display()

		t=w[outline_module]

		t.run(["sed","s/current/voltage/g"])
		t.display()
		#a_out=w.compile()
		#a_out
