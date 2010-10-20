from pyps import workspace, module
with workspace("silber.c","include/adds.c") as w:
	# print out all functions
	w.all.display()
	# get some help
	help(module.unfolding)
	# apply what we have just learnt
	w.fun.transfo.unfolding()
	# select a transformation
	t = w.fun.transfo
	t.display()
	t.flatten_code()
	# this will stop when force_loop_fusion fails
	try:
		while len(t.loops()) > 0:
			l=t.loops()[0]
			l.force_loop_fusion()
			t.suppress_dead_code()
			l.display()
	except:pass
	# if everything is ok ...
	if len(t.loops()) == 1:
		t.forward_substitute(optimistic_clean=True)
		t.display()
		t.common_subexpression_elimination(skip_lhs=False)
		t.suppress_dead_code()
		t.display()
		# greate, saturated arithmeteic !
		help(module.expression_substitution)
		t.expression_substitution(pattern=w.fun.adds.name)
		t.display()
		t.privatize_module()
		t.coarse_grain_parallelization()
		# great ! OMP pragmas
		t.display()
		a_out=w.compile()
