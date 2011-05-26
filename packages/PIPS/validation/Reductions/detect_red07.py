from __future__ import with_statement # this is to work with python2.5
from pyps import *
ws= workspace("detect_red07.c",deleteOnClose=True)
m=ws.fun.main
m.display()
m.reduction_atomization()
m.display(activate="PRINT_CODE_PROPER_REDUCTIONS")
r=ws.compile()
ws.run(r)
ws.close()

