from __future__ import with_statement # this is to work with python2.5
from pyps import workspace, module

with workspace("basics0.c",deleteOnClose=True) as w:
	w.fun.foo.display()
	print "oldcode:\n",w.fun.foo.code
	w.fun.foo.code="""
int foo(int a) {
    return 2*a;
}
"""
	print "newcode:\n",w.fun.foo.code
	w.fun.foo.display()

