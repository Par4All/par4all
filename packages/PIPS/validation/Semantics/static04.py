from __future__ import with_statement
from pyps import module
from validation import vworkspace

with vworkspace() as w:
    w.all_functions.display()
    w.all_functions.display(module.print_code_proper_effects)
    w.all_functions.display(module.print_code_cumulated_effects)
    w.all_functions.display(module.print_code_transformers)
    w.all_functions.display(module.print_code_preconditions)
    w.all_functions.display(module.print_code_regions)
