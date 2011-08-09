from pyps import *

w=workspace("linearize_array05.c")
w.compile()
for f in w.fun:
	f.linearize_array(LINEARIZE_ARRAY_USE_POINTERS=True)
	f.display()
w.compile()
