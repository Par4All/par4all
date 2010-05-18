#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Par4All Builder Class
'''

import sys, os, re, shutil
from p4a_util import *

class p4a_builder():
	def __init__(self, files, output_file, 
		cflags = "", ldflags = "", 
		extra = [], extra_obj = [], 
		cc = None, ld = None, ar = None):
		(base, ext) = os.path.splitext(output_file)
		flags = ""
		if ext == ".o":
			flags = " ".join([ "-c", "-fPIC", cflags ])
			files += extra
		elif ext == ".a":
			flags = " ".join([ "-static", "-fPIC", cflags ])
			files += extra
			files += extra_obj
		elif ext == ".so":
			flags = " ".join([ "-shared", "-fPIC", cflags, ldflags ])
			files += extra
			files += extra_obj
		elif ext == "":
			flags = " ".join([ "-fPIC", cflags, ldflags ])
			files += extra
			files += extra_obj
		else:
			raise p4a_error("unsupported extension for output file: " + output_file)
		args = [ cc, flags, "-o", output_file ]
		args += files
		run(" ".join(args))

if __name__ == "__main__":
	print(__doc__)
	print("This module is not directly executable")

# What? People still use emacs? :-)
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
