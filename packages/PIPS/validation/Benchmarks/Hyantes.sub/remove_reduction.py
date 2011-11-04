from __future__ import with_statement
from pyps import workspace
with workspace("hyantes.c", "options.c") as w:
    w.props.constant_path_effects=False
    f=w["hyantes!do_run_AMORTIZED_DISK"]
    for l in f.all_loops:
        l.reduction_variable_expansion()
    f.display()
