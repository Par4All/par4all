from __future__ import with_statement
from validation import vworkspace
with vworkspace() as w:
    w.all_functions.display()
    print "seems ok"
