import re
import pyps
from pyps import module, workspace, Maker

class ompMaker(Maker):
    ''' A makefile builder for openmp '''
    def get_ext(self):
        return ".omp"+super(ompMaker,self).get_ext()

    def get_makefile_info(self):
        return [("openmp","Makefile.omp")]+super(ompMaker,self).get_makefile_info()


def openmp(m, verbose = False, internalize_parallel_code=True, loop_parallel_threshold_set=False, **props):
    """parallelize module with opennmp"""
    w = m.workspace
    #select most precise analysis
    w.activate(module.must_regions)
    w.activate(module.transformers_inter_full)
    w.activate(module.interprocedural_summary_precondition)
    w.activate(module.preconditions_inter_full)
    w.activate(module.region_chains)
    w.activate(module.rice_regions_dependence_graph)

    w.props.semantics_compute_transformers_in_context = True
    w.props.semantics_fix_point_operator = "derivative"
    w.props.unspaghettify_test_restructuring = True
    w.props.unspaghettify_recursive_decomposition = True
    w.props.aliasing_across_io_streams = False
    w.props.constant_path_effects = False
    w.props.prettyprint_sequential_style = "do"
    w.props.memory_effects_only = False

    m.loop_fusion()

    if loop_parallel_threshold_set:
        m.omp_loop_parallel_threshold_set(**props)
    m.split_initializations(**props)
    #privatize scalar variables
    m.privatize_module(**props)
    #openmp parallelization coarse grain
    if verbose:
        w.props.parallelization_statistics = True
    #custom functions 
    m.coarse_grain_parallelization(**props)
    if internalize_parallel_code:
        m.internalize_parallel_code(**props)
    m.ompify_code(**props)
    m.omp_merge_pragma(**props)
    if verbose:
        m.display(**props)

pyps.module.openmp=openmp
pyps.modules.openmp=lambda m,verbose=False,internalize_parallel_code=True,loop_parallel_threshold_set=False,**props:map(lambda x:openmp(x,verbose,internalize_parallel_code,loop_parallel_threshold_set,**props),m)

