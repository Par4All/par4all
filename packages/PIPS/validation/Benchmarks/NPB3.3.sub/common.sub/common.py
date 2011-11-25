from __future__ import with_statement
from pyps import workspace, module
from glob import glob
with workspace(*(glob("*s.c") + ["wtime.c"]),name='common') as w:
    w.activate(module.must_regions)
    w.all_functions.display()
    w.all_functions.print_code_proper_effects()
    w.all_functions.print_code_cumulated_effects()
    w.all_functions.print_code_transformers()
    w.all_functions.print_code_preconditions()
    w.all_functions.print_code_regions()
