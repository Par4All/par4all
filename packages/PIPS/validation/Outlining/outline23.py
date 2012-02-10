from __future__ import with_statement
from validation import vworkspace
with vworkspace() as w:
    w.fun.ing.display()
    w.fun.ing.outline(module_name="ding",label="here",independent_compilation_unit=True)
    [ f.print_code_regions() for f in w.fun ]
    [ f.print_code() for f in w.fun ]
    print """
***************
Generated files
***************
"""
    for fname in w.save()[0]:
        print "".join(file(fname).readlines())
        print """
***************
"""

