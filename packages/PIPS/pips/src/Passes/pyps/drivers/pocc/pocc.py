
import pyps,os,subprocess,shutil
from re import match


def poccify(self,options="--pluto --pluto-tile",**props):
    ''' This method will try to find out static control parts, then outline
    them in separated functions, pass pocc (polycc) on it, and get it back in 
    PIPS'''
    # Prefix for temporary outlined functions
    prefix = self.name+"_PYPS_SCOP"
    
    # Static control detection must be done specially for pocc (no function call)
    self.clear_pragma()
    self.static_controlize(across_user_calls=False)
#    self.display("print_code_static_control")

    try:
        # Print pragmas
        self.pocc_prettyprinter()

        # Outline scop parts in new modules
        self.pragma_outliner(prefix=prefix,begin="scop",end="endscop")
    
        for m in self.workspace.filter(lambda m:match(prefix,m.name)):
            m.pocc_prettyprinter()
            m.print_code()
                    
            # Get source file for this module
            printed_rc=os.path.join(m.workspace.dirname,m.workspace.cpypips.show("PRINTED_FILE",m.name))
            code_rc=os.path.join(m.workspace.dirname,m.workspace.cpypips.show("C_SOURCE_FILE",m.name))
        
            # Run PoCC !
            #print "Calling PoCC: pocc "+ options + " %s " % (printed_rc)
            p = subprocess.Popen("pocc "+ options + " %s " % (printed_rc), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (sout,serr) = p.communicate()
            if p.returncode != 0:
              raise RuntimeError("Error while trying to call pocc %d.\nstdout:\n%s\nstderr:\n%s\n" % (p.returncode, sout, serr))

            # Get back polycc result with some filtering !
            basename, extension = os.path.splitext(printed_rc)
            pocc_file = basename + ".pocc" + extension
            p = subprocess.call("grep -v '^#include' %s | cpp > %s" % (pocc_file,code_rc),shell=True)
            # Don't know why but we have to force the parser here !
            m.c_parser()
    except:
        raise
    finally:
        #Always clean (remove pragma and inline the results)
        for m in self.workspace.filter(lambda m:match(prefix,m.name)):
            #m.display()        
            m.clear_pragma()
            m.inlining()
            

pyps.module.poccify = poccify

def poccify(self, concurrent=False, **props):
    """  
"""
    for m in self:
        m.poccify()
        

pyps.modules.poccify = poccify



