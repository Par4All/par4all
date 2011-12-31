from __future__ import with_statement # this is to work with python2.5
from validation import vworkspace

with vworkspace() as w:
    w.props.loop_normalize_one_increment=True
    w.all_functions.validate_phases("print_code_preconditions",
            "linearize_loop_range",
            "print_code_preconditions",
            "loop_normalize",
            "print_code_regions")

