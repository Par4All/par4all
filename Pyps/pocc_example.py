from __future__ import with_statement # this is to work with python2.5

# Here we add an example for interacting with PoCC
# it requires that polycc is in the PATH
# we can't run it in the validation because PoCC isn't installed

from pyps import workspace
import pocc

with workspace("pocc_example.c",name="poly",deleteOnCreate=True) as w:
	w.all_functions.poccify()
	w.fun.main.display()


