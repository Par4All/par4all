from __future__ import with_statement
from validation import vworkspace
with vworkspace() as w:
    w.all_functions.loop_normalize()
    w.all_functions.display()
    w.all_functions.partial_eval()
    w.all_functions.display()
