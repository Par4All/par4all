from __future__ import with_statement # this is to work with python2.5
from pyps import workspace

with workspace("simdau0.c", deleteOnClose=True) as w:
	w.props.SAC_SIMD_REGISTER_WIDTH=128
	w.fun.foo.display()
	w.fun.foo.simdizer_auto_unroll()
	w.fun.foo.display()
	w.fun.foo.partial_eval()
	w.fun.foo.suppress_dead_code()
	w.fun.foo.display()

