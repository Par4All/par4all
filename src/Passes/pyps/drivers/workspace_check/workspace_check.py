from __future__ import with_statement # to cope with python2.5
import pyps

class workspace:
	def __init__(self, ws, *args, **kwargs):
		self._cc_ref = kwargs['compiler_ref']
		self._ws = ws

	def post_init(self, sources, **args):
		self._run_ref()

	def _run_ref(self):
		''' Get the output reference using self._cc_ref '''
		rc,out,err = self._ws.compile_and_run(self._cc_ref)
		if rc != 0:
			raise RuntimeError("workspace_check: reference program returned %d: %s" % (rc,err))
		self._out_ref = out

	def check_output(self, compiler):
		rc,out,err = self._ws.compile_and_run(compiler)
		if rc != 0:
			raise RuntimeError("workspace_check: program returned %d: %s" % (rc,err))
		return (out == self._out_ref, out)

	def get_ref(self): return self._out_ref
