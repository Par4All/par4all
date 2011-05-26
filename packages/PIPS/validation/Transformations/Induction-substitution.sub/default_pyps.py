from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace

with vworkspace() as w:
    w.all_functions.validate_phases("print_code_preconditions","induction_substitution")

