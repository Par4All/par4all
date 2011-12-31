from pyps import *

w = workspace('jacobi.c','p4a_stubs.c',deleteOnClose=True)

w.set_property(loop_normalize_one_increment = True,
               loop_normalize_lower_bound=0,
               loop_normalize_skip_index_side_effect=True)

w.all.loop_normalize()
w.all.privatize_module()

w.all.display(With='PRINT_CODE_REGIONS')

w.all.coarse_grain_parallelization()
w.all.display()

w.all.gpu_ify()

launchers= modules(['p4a_kernel_launcher_0', 'p4a_kernel_launcher_1', 'p4a_kernel_launcher_2', 'p4a_kernel_launcher_3', 'p4a_kernel_launcher_4'])
launchers.kernel_load_store()
launchers.gpu_loop_nest_annotate()
launchers.inlining()
...
