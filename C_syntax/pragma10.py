from __future__ import with_statement
from pyps import workspace
with workspace("pragma10.c") as w:
    for f in w.fun:f.display()
