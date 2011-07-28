from __future__ import with_statement # this is to work with python2.5
from pyps import workspace,module
from terapyps import workspace as teraw
from os import system,environ

workspace.delete("addcst")
with teraw("addcst.c", name="addcst", deleteOnClose=True) as w:
	#environ["CHAINS_DEBUG_LEVEL"]="5"
	#environ["PROPER_EFFECTS_DEBUG_LEVEL"]="8"
	for f in w.fun:
		if f.name != 'main':
			f.terapix_code_generation(debug=True)
