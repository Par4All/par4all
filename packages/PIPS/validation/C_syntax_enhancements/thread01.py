from __future__ import with_statement
from pyps import workspace, module
name="thread01"
with workspace(name+".c",name=name,deleteOnCLose=True,deleteOnCreate=True) as w:
    w.fun.main.display()
    w.cu.thread01.display()


