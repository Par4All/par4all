#!/usr/bin/env python
from pyps import workspace
import binary_size

w = workspace("func_size.c", verbose=False, deleteOnClose=True) 
t= w.fun.muladd

(s,c) = t.binary_size()
th_c = 31;
if c < th_c - 10 or c > th_c + 10:
	#This can happen for many reasons (including good ones).
	raise RuntimeError("Instruction count " + str(c) + " is far from the reference instruction count " + str(th_c))

if s < 10 or s > 1000:
	#This can happen for many reasons (including good ones).
	raise RuntimeError("Binary size " + str(s) + " is either very huge or very small")
