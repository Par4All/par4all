from pyps import workspace,module
import terapyps
from os import system
workspace.delete("add")
with workspace(["add.c", "include/terapix_runtime.c" ], cppflags="-I.", name="add") as w:
	for f in w.fun:
		if f.cu  not in [ 'terapix_runtime', 'terasm' ] and f.name != 'main':
			f.terapix_code_generation(debug=True)
