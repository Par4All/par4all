from __future__ import with_statement
from validation import vworkspace

with vworkspace() as w:
    f=w.fun.main
    f.display()
    f.loops("rof").reduction_variable_expansion()
    f.display()

