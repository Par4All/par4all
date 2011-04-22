import re
import pyps

import git

default_excluded_phases = ("code", "unsplit", "print_code")

class git(pyps.workspace):
	''' This workspace will set the workspace files in a git
	repository, tracking modification along passes. '''

	def __init__(self, *args, **kwargs):
		super(git,self).__init__(*args, **kwargs)
		self.set_phase_log_buffer(True)
		self._excluded_phases = kwargs.get("excluded_phases", list())
		self._dirbase = os.path.join(self.tmpdir, kwargs.get("git_dirbase", "git"))
		for p in default_excluded_phases:
			if p not in self._excluded_phases:
				self._excluded_phases.append(p)
		
	def pre_phase(self, phase, module):
		if phase in self._excluded_phases:
			return

	def post_phase(self, phase, module, log):
		if phase in self._excluded_phases:
			return
