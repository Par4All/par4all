from __future__ import with_statement # this is to work with python2.5
# convertion of Semantics/all04.tpips into pyps
# tests usage of capply
from pyps import workspace,module
with workspace('capply01.f',deleteOnClose=True) as w:
	print " Initial code with preconditions for ALL04 after cleanup"
	w.activate(module.transformers_inter_full)
	w.activate("interprocedural_summary_precondition")
	w.activate(module.preconditions_inter_full)
	w.activate(module.print_code_cumulated_effects)

	w.all.display()
	# there is a capply there
	w.all.partial_eval(concurrent=True)
	w.all.display()
	# and there
	w.all.suppress_dead_code(concurrent=True)
	w.all.display()

