from __future__ import with_statement
from validation import vworkspace
import os
with vworkspace() as w:
    w.all_functions.display()
    w.save()
    os.system(" ".join([os.environ.get("PAGER","cat"),os.path.join(w.tmpdirname,w.name+".c")]))
