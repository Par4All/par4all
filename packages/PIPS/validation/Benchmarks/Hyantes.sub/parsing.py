from __future__ import with_statement
from pyps import workspace
with workspace("hyantes.c", "options.c") as w:
    w.all_functions.display()
