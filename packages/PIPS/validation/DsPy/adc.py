from __future__ import with_statement
from pyps import *
from openmp import *

class mycompiler(gccCompiler,ompCompiler):
	pass

with workspace("adc.c","dac.c", "u.c", "tools.c", cppflags="-DADC_MAIN") as w:
	w.all_functions.openmp(verbose=True)
	w.compile(compiler=mycompiler(LDFLAGS="-lm"))
