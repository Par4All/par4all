from pyps import *
import re

# filter to select only lauchers
launcher_re = re.compile("^p4a_kernel_launcher_.*")
def launcher_filter(module):
	return launcher_re.match(module.name)

# create the pips workspace
options=["MUST_REGIONS","TRANSFORMERS_INTER_FULL","INTERPROCEDURAL_SUMMARY_PRECONDITION","PRECONDITIONS_INTER_FULL"]
w = workspace(["jacobi.c","p4a_stubs.c"],options)

# apply several transformations to all modules
w.all().loop_normalize(one_increment=True,lower_bound=0,skip_index_side_effect=True)
w.all().privatize_module()

# display modules with region annotations
w.all().display(With="PRINT_CODE_REGIONS")

w.all().coarse_grain_parallelization()
w.all().display()

w.all().gpu_ify()

# select only some modules from the workspace
launchers=w.all(launcher_filter)
# manipulate them as first level objects
launchers.kernel_load_store()
launchers.display()

launchers.gpu_loop_nest_annotate()
launchers.inlining()

# save resulting code in a given directory
outdir="p4a_gen"
w.save(indir=outdir)
