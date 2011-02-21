from __future__ import with_statement # this is to work with python2.5
from pyps import workspace

with workspace("rn_op.c") as w:
	ms = w.fun.sum_f
	ms.display()

	#Try to convert only the operator (float) + (float)
	ms.rename_operator(suffixes="f", ops="plus")
	ms.display()

	#Try to convert all operators that work on float
	ms.rename_operator(suffixes="f")
	ms.display()

	#Try to convert integers
	ms.rename_operator(suffixes="i")
	ms.display()

	#Substitution of the loop range
	ms.rename_operator(suffixes="i", rewrite_do_loop_range=True)
	ms.display()

	#Another function
	ms = w.fun.muladd_f
	ms.rename_operator()
	ms.display()

	#And complex
	ms = w.fun.sum_c
	ms.rename_operator()
	ms.display()

