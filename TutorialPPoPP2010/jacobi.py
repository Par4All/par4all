from pyps import *

# create the pips workspace
w = workspace(['jacobi.c','p4a_stubs.c'])

w.activate('MUST_REGIONS')
w.activate('TRANSFORMERS_INTER_FULL')
w.activate('INTERPROCEDURAL_SUMMARY_PRECONDITION')
w.activate('PRECONDITIONS_INTER_FULL')

w.set_property( loop_normalize_1_increment = True,
                loop_normalize_lower_bound=0,
                loop_normalize_skip_index_side_effect = True
)

# apply several transformations to all modules
w.all.loop_normalize()
w.all.privatize_module()

# display modules with region annotations
w.all.display(With='PRINT_CODE_REGIONS')

w.all.coarse_grain_parallelization()
w.all.display()

w.all.gpu_ify()

# select only some modules from the workspace
launchers= ['p4a_kernel_launcher_0', 'p4a_kernel_launcher_1', 'p4a_kernel_launcher_2', 'p4a_kernel_launcher_3', 'p4a_kernel_launcher_4']
map(lambda name: w[name].kernel_load_store(), launchers)
map(lambda name: w[name].gpu_loop_nest_annotate(),launchers)
map(lambda name: w[name].inlining(),launchers)

# save resulting code in a given directory
outdir='p4a_gen'
w.save(indir=outdir)
