from pyps import *
import sys
w = workspace(["basics0.c"],verbose=False)
print >> sys.stderr, "hello"
for f in w.fun:f.display()
w.close()
