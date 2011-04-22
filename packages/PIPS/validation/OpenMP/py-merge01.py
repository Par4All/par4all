from __future__ import with_statement
from pyps import *
import openmp

with workspace("merge01.c") as w:
	w.props.memory_effects_only=False
	w.all_functions.openmp(verbose=True)
