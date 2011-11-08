from __future__ import with_statement # this is to work with python2.5
from validation import vworkspace
from pyps import module

with vworkspace() as w:
    w.all_functions.display(module.print_code_preconditions)
    w.all_functions.linearize_loop_range()
    w.all_functions.display(module.print_code_preconditions)
    w.all_functions.loop_normalize(one_increment=True)
    w.all_functions.display(module.print_code_regions)

