from pyps import *
import re

program = "jacobi01"

# Just in case it existed before:
workspace.delete(program)
with workspace([ program + ".c",	"include/p4a_stubs.c" ], name = program) as w:
	w.activate(module.transformers_inter_full)
	w.activate(module.interprocedural_summary_precondition)
	w.activate(module.preconditions_inter_full)
	w.activate(module.must_regions)
	w.props.PRETTYPRINT_STATEMENT_NUMBER = True
	w.props.FOR_TO_DO_LOOP_IN_CONTROLIZER = True
	w.props.SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT = True
	w.props.SEMANTICS_FIX_POINT_OPERATOR = "derivative"
	       # Some temporary hack to have this parallelized loop:
## 	       /* Erase the memory, in case the image is not big enough: */
## #pragma omp parallel for private(j)
##    for(i = 0; i <= 500; i += 1)
## #pragma omp parallel for
##       for(j = 0; j <= 500; j += 1)
##          space[i][j] = 0;                                               /*0045*/
	w.props.ALIASING_ACROSS_IO_STREAMS = False


	# Skip module name of P4A runtime:
	skip_p4a_runtime_and_compilation_unit_re = re.compile("P4A_.*|.*!")
	def is_not_p4a_runtime(module):
		#print module.name
		return not skip_p4a_runtime_and_compilation_unit_re.match(module.name)
	
	mn = w.filter(is_not_p4a_runtime)
	
	
	#for i in mn:
	#	print i.name
	
	mn.loop_normalize(
		# Loop normalize for the C language and GPU friendly
		LOOP_NORMALIZE_ONE_INCREMENT = True,
		LOOP_NORMALIZE_LOWER_BOUND = 0,
		# It is legal in the following by construction:
		LOOP_NORMALIZE_SKIP_INDEX_SIDE_EFFECT = True)
	
	mn.privatize_module()
	
	mn.display(With="PRINT_CODE_REGIONS")
	
	
	# mn.localize_declaration()
	
	# mn.display(With="PRINT_CODE_PRECONDITIONS")
	
	mn.coarse_grain_parallelization()
	mn.display()
	
	mn.gpu_ify()
	mn.display()
	
	#setproperty KERNEL_LOAD_STORE_ALLOCATE_FUNCTION "P4A_ACCEL_MALLOC"
	#setproperty KERNEL_LOAD_STORE_DEALLOCATE_FUNCTION "P4A_ACCEL_FREE"
	#setproperty KERNEL_LOAD_STORE_LOAD_FUNCTION "P4A_COPY_TO_ACCEL"
	#setproperty KERNEL_LOAD_STORE_STORE_FUNCTION "P4A_COPY_FROM_ACCEL"
	
	# Isolate kernels by using the fact that all the generated kernels have
	# their name beginning with "p4a_":
	kernel_launcher_filter_re = re.compile("p4a_kernel_launcher_.*[^!]$")
	kernel_launchers = w.filter(lambda m: kernel_launcher_filter_re.match(m.name))
	
	#kernels.display()
	# Add communication around all the call site of the kernels:
	kernel_launchers.kernel_load_store()
	kernel_launchers.display()
	
	kernel_launchers.gpu_loop_nest_annotate()
	kernel_launchers.display()
	
	# Inline back the kernel into the wrapper, since CUDA can only deal with
	# local functions if they are in the same file as the caller (by inlining
	# them, by the way... :-) )
	kernel_filter_re = re.compile("p4a_kernel_\\d+$")
	kernels = w.filter(lambda m: kernel_filter_re.match(m.name))
	kernels.inlining()
	
	# Display the wrappers to see the work done:
	kernel_wrapper_filter_re = re.compile("p4a_kernel_wrapper_\\d+$")
	kernel_wrappers = w.filter(lambda m: kernel_wrapper_filter_re.match(m.name))
	kernel_wrappers.display()
	
	# Instead, do a global loop normalization above:
	#kernels.loop_normalize()
	#kernels.use_def_elimination()
	#display PRINTED_FILE[p4a_kernel_launcher_0,p4a_kernel_launcher_1,p4a_kernel_launcher_2,p4a_kernel_launcher_3,p4a_kernel_launcher_4]
	
	#w.all.suppress_dead_code()
	#w["main"].display()
	
	w["main"].prepend_comment(PREPEND_COMMENT = "// Prepend here P4A_init_accel")
	
	# Unsplit resulting code
	w.all.unsplit()
	
	# Generating P4A code:
	#os.system("cd " + program + ".database; p4a_post_processor.py Src/*.c")
