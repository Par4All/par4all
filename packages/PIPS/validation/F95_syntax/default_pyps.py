from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace

with vworkspace() as w:
    w.all_functions.display()
    w.save() # important to test unsplit in F95 since it's quite unstable at that time
    

