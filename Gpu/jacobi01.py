from __future__ import with_statement # this is to work with python2.5
from pyps import *
import re

program = "jacobi01"

# Just in case it existed before:
workspace.delete(program)
with workspace(program + ".c",	"include/p4a_stubs.c", name = program, deleteOnClose=False) as w:
	w.activate(module.transformers_inter_full)
	w.activate(module.interprocedural_summary_precondition)
	w.activate(module.preconditions_inter_full)
	w.activate(module.must_regions)
	w.props.SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT = True
	w.props.SEMANTICS_FIX_POINT_OPERATOR = "derivative"
	w.props.UNSPAGHETTIFY_TEST_RESTRUCTURING = True
	w.props.UNSPAGHETTIFY_RECURSIVE_DECOMPOSITION = True
	w.props.FOR_TO_DO_LOOP_IN_CONTROLIZER = True
	w.props.KERNEL_LOAD_STORE_VAR_PREFIX = ""
	       # Some temporary hack to have this parallelized loop:
## 	       /* Erase the memory, in case the image is not big enough: */
## #pragma omp parallel for private(j)
##    for(i = 0; i <= 500; i += 1)
## #pragma omp parallel for
##       for(j = 0; j <= 500; j += 1)
##          space[i][j] = 0;                                               /*0045*/
	w.props.ALIASING_ACROSS_IO_STREAMS = False

	w.props.CONSTANT_PATH_EFFECTS = False
	w.props.PRETTYPRINT_STATEMENT_NUMBER = True

	# Skip module name of P4A runtime:
	skip_p4a_runtime_and_compilation_unit_re = re.compile("P4A_.*|.*!")
	def is_not_p4a_runtime(module):
		#print module.name
		return not skip_p4a_runtime_and_compilation_unit_re.match(module.name)

	mn = w.filter(is_not_p4a_runtime)

        w.fun.iteration.display()
