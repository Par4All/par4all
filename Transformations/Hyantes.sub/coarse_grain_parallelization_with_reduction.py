from __future__ import with_statement
from pyps import workspace, module
with workspace("hyantes.c", "options.c") as w:
    w.activate(module.must_regions)
    w.props.constant_path_effects=False
    f=w["hyantes!do_run_AMORTIZED_DISK"]
    f.coarse_grain_parallelization_with_reduction()
    f.display()
