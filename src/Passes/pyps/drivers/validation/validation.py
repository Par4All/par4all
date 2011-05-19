import pyps,os,sys

class vworkspace(pyps.workspace): 
    ''' This workspace is intended to handle some special PIPS validation 
    suff'''
    
    def __init__(self, *sources, **kwargs):
        """init a workspace from (optional) sources for validation
           name will be gather from WSPACE and sources from FILE environment variables 
        """

        file = os.getenv('FILE')
        if file != None and not os.path.isfile(file):
            sys.stderr.write("Error, file " + file + " doesn't exists !!")
            file = None
            
        if file == None :
            # Try to recover source corresponding to script filename
           basename = os.path.splitext(sys.argv[0])[0]
           file = basename + '.c'
           if not os.path.isfile(file):
               file = basename + '.f'
           if not os.path.isfile(file):
               file = basename + '.f95'
           if not os.path.isfile(file):
               raise RuntimeError('''No source files ! Please define FILE environment 
                                    variable or provide %s.{c,f,f95}''' % (file) )
            
        # this workspace is intended to be run with WSPACE and FILE 
        # environment variable defined
        wspace = os.getenv("WSPACE")
        if wspace == None :
           sys.stderr.write("WSPACE environment variable has to be defined\n")
           wspace = os.path.splitext(sys.argv[0])[0]

        super(vworkspace, self).__init__(file,
                                         *sources,
                                         name=wspace,
                                         deleteOnClose=True, 
                                         deleteOnCreate=True,
                                         **kwargs)


def validate_phases(self,*phases,**kwargs):
    display_after= kwargs.setdefault("display_after",  True)
    display_initial= kwargs.setdefault("display_initial",  True)
    if display_initial:
        print "//"
        print "// Initial code for module " + self.name
        print "//"
        self.display()
    for phase in list(phases):
        if "print_code_" in phase:
            print "//"
            print "// Display " + phase + " for module " + self.name
            print "//"
            self.display(phase);
        else:
            getattr(self,phase)()
            if display_after:
                print "//"
                print "// Code after " + phase + " for module " + self.name
                print "//"
                self.display()
        
pyps.module.validate_phases = validate_phases
        
        
def validate_phases(self, *phases,**kwargs):
    display_initial= kwargs.setdefault("display_initial",  False)
    display_after= kwargs.setdefault("display_after",  True)
    if display_initial:
        for m in self:
            print "//"
            print "// Initial code for module " + m.name
            print "//"
            m.display()
    for phase in list(phases):
        for m in self:
            m.validate_phases(phase,display_after=display_after,display_initial=False)

pyps.modules.validate_phases = validate_phases
