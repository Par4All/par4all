from __future__ import with_statement # to cope with python2.5
import pyps

class workspace(pyps.workspace):
    """ A workspace that checks program output after each phase"""
    def __init__(self, *sources, **kwargs):
        super(workspace,self).__init__(*sources, **kwargs)
        self.__ref_maker = kwargs.get('ref_maker',pyps.Maker())
        self.__ref_argv = kwargs.get('ref_argv',[])
        self.__ref_output=None
        self._enable_check = True

    def __compile_and_run(self):
        if "main" in self.fun:
            a_out = self.compile(self.__ref_maker)
            (_,cout,_) = self.run(a_out,self.__ref_argv)
            return cout

    def pre_phase(self,phase,module):
        """ generate reference if needed """
        if self._enable_check and self.__ref_output == None:
            self.__ref_output=self.__compile_and_run()
        super(workspace,self).pre_phase(phase,module)

    def post_phase(self,phase,module):
        """ check resulting code after the apply """
        super(workspace,self).post_phase(phase,module)
        if self._enable_check:
            output=self.__compile_and_run()
            if output != self.__ref_output:
                raise RuntimeError("check by workspace check failed after %s on %s\n" % (phase , module ))

