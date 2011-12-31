from __future__ import with_statement
from pyps import workspace
wname = "float02"
with workspace(wname+".c",name=wname,deleteOnCLose=True, deleteOnCreate=True) as w:
    w.fun.main.display()
