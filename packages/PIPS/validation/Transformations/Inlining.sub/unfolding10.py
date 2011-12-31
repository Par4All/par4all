from __future__ import with_statement
import pyps
pyps.workspace.delete("unfolding10")
with pyps.workspace("unfolding10.c",name="unfolding10",deleteOnClose=True) as w:
    w.fun.main.display()
    w.fun.main.outline(label=w.fun.main.loops(1).loops(0).label,module_name="new")
    w.fun.main.display()
    w.fun.new.display()
    w.fun.new.unfolding()
    w.fun.new.display()
