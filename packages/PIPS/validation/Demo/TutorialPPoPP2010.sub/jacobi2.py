from pyps import *
import re

launcher_re = re.compile("^p4a_kernel_launcher_.*")
def launcher_filter(module):
	return launcher_re.match(module.name)

w = workspace("jacobi.c","p4a_stubs.c",deleteOnClose=True)

w.all.loop_normalize(one_increment=True,lower_bound=0,skip_index_side_effect=True)
w.all.privatize_module()

w.all.display(activate=module.print_code_regions)

w.all.coarse_grain_parallelization()
w.all.display()

w.all.gpu_ify()

# select only some modules from the workspace
launchers=w.all(launcher_filter)
# manipulate them as first level objects
launchers.kernel_load_store()
launchers.display()

launchers.gpu_loop_nest_annotate()
launchers.inlining()
...
