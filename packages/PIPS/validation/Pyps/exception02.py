from __future__ import with_statement # this is to work with python2.5
from validation import vworkspace

with vworkspace() as w:
    try:
        w.all_functions.scalarization(threshold=1,concurrent=True) # called in parallel with a bad argument, should raise an exception
        print"this should never be printed"
    except:
        print "an exception was raised, everything is ok"
