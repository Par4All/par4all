from __future__ import with_statement # this is to work with python2.5
#!/usr/bin/env python
from pyps import *
with workspace("properties3.c",deleteOnClose=True) as ws:
    # properties unit test
    try:
        ws.cu.truc="muche"
    except AttributeError:
        print "error caught"
        pass
    try:
        print ws.cu.loop_label
    except NameError:
        print "error caught"
        pass

    c=ws.cu.properties3

    print ws.cu.properties3.name

    filter(lambda x: x.display() if x.name == "properties3!" else None , ws.cu)

    print "yes" if c.name in ws.cu else "no"
    print "yes" if "properties3" in ws.cu else "no"

