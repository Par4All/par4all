
import pyps,os,subprocess,shutil
from re import match


def poccify(self,**props):
    ''' This method will try to find out static control parts, then outline
    them in separated functions, pass pocc (polycc) on it, and get it back in 
    PIPS'''

    # Static control detection must be done specially for pocc (no function call)
    self.clear_pragma()
    self.static_controlize(across_user_calls=False)

    try:
        # Print pragmas
        self.pocc_prettyprinter()

        # Outline scop parts in new modules
        self.scop_outliner(scop_prefix="PYPS_SCOP")
    
        # FIXME should only poccify previously outlined modules !! 
        # need work on pips side
        for m in self.workspace.filter(lambda m:match("PYPS_SCOP",m.name)):
            m.print_code()
                    
            # Get source file for this module
            code_rc=os.path.join(m.workspace.dirname,m.workspace.cpypips.show("C_SOURCE_FILE",m.name))
        
            # Run PoCC !
            # FIXME : args should be user defined (but it used affect the output filename :-(
            print "Calling pocc + pocc --pluto --pluto-tile %s " % (code_rc)
            p = subprocess.Popen("pocc --pluto --pluto-tile %s " % (code_rc), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (sout,serr) = p.communicate()
            if p.returncode != 0:
              raise RuntimeError("Error while trying to call pocc %d.\nstdout:\n%s\nstderr:\n%s\n" % (p.returncode, sout, serr))

            # Get back polycc result with some filtering !
            basename, extension = os.path.splitext(code_rc)
            pocc_file = basename + ".pocc" + extension
            p = subprocess.call("grep -v '^#include' %s | grep -v '^#define' | grep -v '^/\*' > %s" % (pocc_file,code_rc),shell=True)

    except:
        raise
    #finally:
        # Always clean (remove pragma and inline the results)
        #for m in self.workspace.filter(lambda m:match("PYPS_SCOP",m.name)):        
            #m.remove_pocc_pragma()
            #m.inlining()

pyps.module.poccify = poccify

def poccify(self, concurrent=False, **props):
    """  
"""
    for m in self:
        m.poccify()
        

pyps.modules.poccify = poccify



