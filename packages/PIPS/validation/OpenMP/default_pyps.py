from __future__ import with_statement
import openmp
from validation import vworkspace

with vworkspace() as w:
    w.all_functions.validate_phases("openmp")
    w.all_functions.display()

