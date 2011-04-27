from __future__ import with_statement
from pyps import *
import openmp
from os.path import basename, splitext
from sys import argv

(filename,_)=splitext(basename(argv[0]))
with workspace(filename[3:]+".c") as w:
	w.props.memory_effects_only=False
	w.all_functions.openmp(verbose=True)
