from __future__ import with_statement
from pyps import workspace, module
with workspace("hyantes.c", "options.c") as w:
    w.activate(module.must_regions)
    w.props.constant_path_effects=False
    w.props.prettyprint_sequential_style="do"
    w.all_functions.flag_parallel_reduced_loops_with_openmp_directives()
    w.all_functions.display()
