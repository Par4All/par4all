from __future__ import with_statement # this is to work with python2.5
from pyps import workspace, module
with workspace("string06.c",deleteOnClose=True) as w:
    map(module.display,w.fun)
