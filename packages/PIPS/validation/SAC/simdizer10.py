# test vectorization when the vector size is equal to that of a member : no vectorization in that case
from __future__ import with_statement
from pyps import workspace
with workspace("simdizer10.c", "include/SIMD.c") as w:
    try:
        w.fun.foo_l.simdizer(sac_simd_register_width=64)
        print "should not be reached"
    except:
	    w.fun.foo_l.display()
	    w.fun.foo_l.simdizer(sac_simd_register_width=128)
	    w.fun.foo_l.display()
