from __future__ import with_statement # this is to work with python2.5
from pyps import workspace

with workspace("pragma1.c",deleteOnClose=True) as w:
    print w.fun.foo.loops(0).pragma

