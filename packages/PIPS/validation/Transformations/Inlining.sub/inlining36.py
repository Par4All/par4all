from __future__ import with_statement
from validation import vworkspace as mehdi_nice_workspace

with mehdi_nice_workspace() as w:
    w.fun.idhem.inlining(comment_origin=True)
    w.fun.mehdi.display()

