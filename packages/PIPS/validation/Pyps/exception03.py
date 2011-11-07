from __future__ import with_statement
#from pyps import module
from validation import vworkspace
with vworkspace() as w:
    f=w.fun.usud
    print "** original code **"
    f.display()
    try:
        f.loops('rof').strip_mine(factor=4,kind=0)
        print "(0.0) strip mining ok"
    except:
        print "strip mining failed, try another strategy"
        chk0=w.checkpoint()
        print "(1.0) checkpointing ok ..."
        try:
            f.loop_normalize()
            print "(1.1) loop normalization ok ..."
            f.partial_eval()
            print "(1.2) partial eval ok ..."
            f.loops('rof').strip_mine(factor=4,kind=0)
            print "(1.3) strip mining ok ..."
            f.isolate_statement(label='rof')
            print "(1.4) isolate statement ok ... It should not be the case!"
        except:
            print "failure at some point, restore previous context"
            w.restore(chk0)
            print "(2.0) restore ok ..."
            f.loops('rof').unroll(rate=4)
            print "(2.1) loop unroll ok ..."
            f.partial_eval()
            print "(2.2) partial eval ok ..."
            pass
    f.display()
