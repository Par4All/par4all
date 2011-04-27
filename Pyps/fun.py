from __future__ import with_statement # this is to work with python2.5
#!/usr/bin/env python
from pyps import *
with workspace("properties3.c",deleteOnClose=True) as ws:
    # properties unit test
    try:
        ws.fun.truc="muche"
    except AttributeError:
        print "error caught"
        pass
    try:
        print ws.fun.loop_label
    except NameError:
        print "error caught"
        pass

    c=ws.fun.convol

    print ws.fun.convol.name

    filter(lambda x: x.display() if x.name == "convol" else None , ws.fun)

    print "yes" if c.name in ws.fun else "no"
    print "yes" if "convol" in ws.fun else "no"

