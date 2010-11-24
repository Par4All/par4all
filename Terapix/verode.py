from pyps import workspace,module
import terapyps
from os import system
workspace.delete("verode")
with workspace(["verode.c", "include/terapix_runtime.c" ], cppflags="-I.", name="verode", deleteOnClose=True) as w:
	for f in w.fun:
		if f.cu  not in [ 'terapix_runtime', 'terasm' ] and f.name != 'main':
			f.terapix_code_generation(debug=True)

