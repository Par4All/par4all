from pyps import *

w = workspace(["basics0.c"])
w.fun.foo.inlining(callers="malabar")
w.checkpoint()
w.fun.foo.inlining(callers="bar")
w.fun.megablast.display()
w.restore()
w.fun.megablast.display()
w.fun.foo.inlining(callers="megablast")
w.fun.megablast.display()

