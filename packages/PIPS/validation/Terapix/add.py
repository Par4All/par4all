from __future__ import with_statement # this is to work with python2.5
from pyps import workspace,module
from terapyps import workspace as teraw
from os import system
workspace.delete("add")
with teraw("add.c", name="add", deleteOnClose=True) as w:
	for f in w.fun:
		if f.name != 'main':
			f.terapix_code_generation(debug=True)
