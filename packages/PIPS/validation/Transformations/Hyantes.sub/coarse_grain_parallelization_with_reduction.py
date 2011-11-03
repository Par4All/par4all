from __future__ import with_statement
from pyps import workspace, module
with workspace("hyantes.c", "options.c") as w:
    w.activate(module.must_regions)
    w.props.constant_path_effects=False
    f=w["hyantes!do_run_AMORTIZED_DISK"]
    f.flag_parallel_reduced_loops_with_openmp_directives()
    f.display()
