from __future__ import with_statement # this is to work with python2.5
#!/usr/bin/env python
from pyps import *
with workspace("properties3.c",deleteOnClose=True) as ws:
    conv=ws.fun.convol
    # properties unit test
    try:
        ws.props.tru="muche"
    except NameError:
        print "error caught"
        pass
    try:
        ws.props.loop_label=1
    except RuntimeError:
        print "error caught"
        pass

    print ws.props.OUTLINE_MODULE_NAME

    print filter(lambda (x,y): x if x == "MAXIMUM_USER_ERROR" else None , ws.props)

    print "yes" if "MAXIMUM_USER_ERROR" in ws.props else "no"

