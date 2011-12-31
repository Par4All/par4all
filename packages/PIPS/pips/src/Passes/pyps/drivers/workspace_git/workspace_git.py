import os,git,glob
import pyps

class workspace(pyps.workspace):
    ''' This workspace will set the workspace files in a git
    repository, tracking modification along passes. '''

    __default_excluded_phases = set(["code", "unsplit", "print_code"])

    def __init__(self, *args, **kwargs):
        super(workspace,self).__init__(*args, **kwargs)
        self.phase_log_buffer=True
        self.__excluded_phases = kwargs.get("excluded_phases", set())
        self.__dirbase = os.path.join(self.tmpdirname, "git")
        self.__excluded_phases.update(self.__default_excluded_phases)

        self.__gitdir = os.path.join(self.__dirbase,".git")
        self.__g = git.repo.Repo.create(self.__gitdir)

        self.__git_do_commit("Initial workspace")
    
    def __get_module_file(self, module):
        return os.path.join(self.__dirbase, module.name+".c")

    def __git_do_commit(self,msg):
        self.save(self.__dirbase)
        # We should do something better, but I can't understand how this version of python-git really works...
        cfiles = " ".join(map(os.path.basename, glob.glob(os.path.join(self.__dirbase, "*.c"))))
        os.system("git --git-dir=%s --work-tree=%s add %s" % (self.__gitdir,self.__dirbase, cfiles))
        os.system("git --git-dir=%s --work-tree=%s commit -m \"%s\"" % (self.__gitdir, self.__dirbase, msg))

    def pre_phase(self, phase, module):
        super(workspace,self).pre_phase(phase,module)
        if not phase in self.__excluded_phases:
            module.workspace.cpypips.open_log_buffer()

    def post_phase(self, phase, module):
        super(workspace,self).post_phase(phase,module)
        if not phase in self.__excluded_phases:
            log = module.workspace.cpypips.get_log_buffer()
            module.workspace.cpypips.close_log_buffer()

            source = self.__get_module_file(module)
            commit_msg = "Phase %s on module %s.\n%s" % (phase,module.name,log)
            self.__git_do_commit(commit_msg)
