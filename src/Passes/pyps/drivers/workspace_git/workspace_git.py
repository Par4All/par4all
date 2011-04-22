import os,git,glob
import pyps,pypips

class workspace(pyps.workspace):
	''' This workspace will set the workspace files in a git
	repository, tracking modification along passes. '''

	default_excluded_phases = ("code", "unsplit", "print_code")

	def __init__(self, *args, **kwargs):
		super(workspace,self).__init__(*args, **kwargs)
		self.set_phase_log_buffer(True)
		self._excluded_phases = kwargs.get("excluded_phases", list())
		self._dirbase = os.path.join(self.dirname, "Src")
		for p in self.default_excluded_phases:
			if p not in self._excluded_phases:
				self._excluded_phases.append(p)

		self._gitdir = os.path.join(self._dirbase,".git")
		self._g = git.repo.Repo.create(self._gitdir)

		self.cpypips.apply("UNSPLIT","%ALL")
		self._git_do_commit("Initial workspace")
	
	def _get_module_file(self, module):
		return os.path.join(self._dirbase, module.name+".c")

	def _git_do_commit(self,msg):
		# We should do something better, but I can't understand how this version of python-git really works...
		cfiles = " ".join(map(os.path.basename, glob.glob(os.path.join(self._dirbase, "*.c"))))
		os.system("git --git-dir=%s --work-tree=%s add %s" % (self._gitdir,self._dirbase, cfiles))
		os.system("git --git-dir=%s --work-tree=%s commit -m \"%s\"" % (self._gitdir, self._dirbase, msg))

	def pre_phase(self, phase, module):
		super(workspace,self).pre_phase(phase,module)
		if not phase in self._excluded_phases:
			module._ws.cpypips.open_log_buffer()

	def post_phase(self, phase, module):
		super(workspace,self).post_phase(phase,module)
		if not phase in self._excluded_phases:
			log = module._ws.cpypips.get_log_buffer()
			module._ws.cpypips.close_log_buffer()

			source = self._get_module_file(module)
			commit_msg = "Phase %s on module %s.\n%s" % (phase,module.name,log)
			self.cpypips.apply("UNSPLIT","%ALL")
			self._git_do_commit(commit_msg)
