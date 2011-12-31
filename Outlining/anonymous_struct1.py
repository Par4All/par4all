from __future__ import with_statement # this is to work with python2.5
from validation import vworkspace

with vworkspace() as w:
    w.all_functions.outline(module_name= "kernel",label="kernel",independent_compilation_unit=True)
    w.all_functions.display()
    w.fun.main.proper_effects()
    w.compile()
