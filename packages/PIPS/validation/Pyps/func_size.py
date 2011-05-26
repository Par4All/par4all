#!/usr/bin/env python
from pyps import workspace
import binary_size

w = workspace("func_size.c", verbose=False, deleteOnClose=True) 
t= w.fun.muladd

(s,c) = t.binary_size()
# reference is on Linux 32bit Debian Sid
th_c = 31;
if c < th_c - 16 or c > th_c + 16:
	#This can happen for many reasons (including good ones).
	print "Instruction count " + str(c) + " is far from the reference instruction count " + str(th_c)
else:
	print "Instruction count ok"

if s < 10 or s > 1000:
	#This can happen for many reasons (including good ones).
	print "Binary size " + str(s) + " is either very huge or very small"
else:
	print "Binary size ok"
