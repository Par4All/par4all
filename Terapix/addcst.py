from pyps import workspace,module
from terapyps import workspace as teraw
from os import system
workspace.delete("addcst")
with workspace("addcst.c", name="addcst", parents=[teraw],deleteOnClose=True) as w:
	for f in w.fun:
		if f.name != 'main':
			f.terapix_code_generation(debug=True)
