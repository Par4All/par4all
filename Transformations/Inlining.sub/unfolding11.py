from __future__ import with_statement
import pyps
pyps.workspace.delete("unfolding11")
with pyps.workspace("unfolding11.c",name="unfolding11",deleteOnClose=True) as w:
    w.fun.main.display()
    w.fun.main.unfolding()
    w.fun.main.display()
