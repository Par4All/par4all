from __future__ import with_statement
from pyps import workspace, module
name="time05"
with workspace(name+".c",name=name,deleteOnCLose=True,deleteOnCreate=True) as w:
    w.fun.main.display(activate=module.print_code_proper_effects)


