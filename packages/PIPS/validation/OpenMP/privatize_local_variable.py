from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace

with vworkspace() as w:
    w.props.memory_effects_only = False
    w.props.constant_path_effects = False
    w.props.prettyprint_sequential_style = "do"

    w.all_functions.privatize_module()

    w.all_functions.validate_phases("coarse_grain_parallelization","ompify_code","omp_merge_pragma")

