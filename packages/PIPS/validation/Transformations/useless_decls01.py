from __future__ import with_statement
from pyps import workspace

with workspace("useless_decls01.c") as w:
    w.fun.main.display()
    w.fun.main.clean_declarations()
    w.fun.main.display()
