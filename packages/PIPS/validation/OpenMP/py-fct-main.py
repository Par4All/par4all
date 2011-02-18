from __future__ import with_statement
from pyps import *
import openmp

with workspace("fct-main.c") as w:
	w.all_functions.openmp(verbose=True)
