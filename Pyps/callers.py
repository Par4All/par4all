from pyps import *
with workspace(["callers.c"]) as w:
	for c in w.fun.main.callees:print c.name
	w.fun.tenant.callers.display()

