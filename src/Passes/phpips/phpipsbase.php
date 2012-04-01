<?php

include('phpips');

# initialize pipslibs when module is loaded
phpips.atinit()

class Maker {

    function __init__() {
        ''' Loading attribute header and rules '''
        $$this.headers=""
        $this.rules=""
        $this.ext = $this.get_ext()
        $makefile_info = $this.get_makefile_info()
        foreach($mi in makefile_info) {
            $atr = $this.__get_makefile_attributes(mi[1],mi[0])
            $this.headers += atr[0]
            $this.rules += atr[1]
        }
    }
    
    function makefile() {
        """ retrieve the name of the generated Makefile """
        return "Makefile"+$this.ext
    }


    function generate($path,$sources,$cppflags="",$ldflags="") {
        """ create a Makefile with additional flags if given """
        mk="SOURCES="+" ".join(sources)+"\n"+\
            $headers+"\n"+\
            "CPPFLAGS+="+cppflags+"\n"+\
            "LDFLAGS+="+ldflags+"\n"+\
            $this.rules
        return basename($path,$makefile))),[]
    }

    function __get_makefile_attributes(makefile,makefiledir) {
        l = file2string(get_runtimefile(makefile,makefiledir)).split("##pipsrules##")
        return (l[0],l[1])
    }

    function get_ext() {
        """ retrieve makefile extension, specialize it to differentiate your makefile from others"""
        return ""
    }

    function get_makefile_info() {
        """ retrieve the list of makefile from which the final makefile is generated"""
        return [("phpipsbase","Makefile.base") ]
    }

class loop {
    """do-loop from a module"""

    function __init__($module,$label) {
        """ bind a loop to its module"""
        $this.__module=module
        $this.__label=label
        $this.__ws=module.workspace
    }

    
    function label() {
        """loop label, as seen in the source code"""
        return $this.__label
    }

    
    function module() {
        """module containing the loop"""
        return $this.__module
    }

    
    function pragma() {
        return $this.__ws.cphpips.loop_pragma(.__module.name,$this.__label)
    }

    
    function workspace() {
        """workspace containing the loop"""
        return $this.__ws
    }

    function display() {
        """display loop module"""
        $this.module.display()
    }

    function loops($label=None) {
        """get outer loops of this loop"""
        loops=$this.__ws.cphpips.module_loops(.module.name,$this.label)
        if label!=None: return $this.loops()[label]
        else: return [ loop(.module,l) foreach($l in str.split(loops," ") ] if loops else []
    }
        
}

class module { 

    function __init__($ws,$name,$source="") {
        """ binds a module to its workspace"""
        $this.__name=name
        $this.__source=source
        $this.__ws=ws
        $this.re_compilation_units = re.compile("^.*!$")
        $this.re_static_function = re.compile("^.*!.+$")
    }
        
    
    function cu() {
        """compilation unit"""
        return $this.__ws.cphpips.compilation_unit_of_module(.name)[:-1]
    }

    
    function workspace() {
        return $this.__ws
    }

    
    function name() {
        """module name"""
        return $this.__name
    }

    
    function language() {
        return $this.__ws.cphpips.get_module_language(.name);
    }

    function compilation_unit_p() {
        return $this.re_compilation_units.match(.name)
    }

    function static_p() {
        return $this.re_static_function.match(.name)

    function edit(editor=os.getenv("EDITOR","vi")) {
        """edits module using given editor
           does nothing on compilation units ...
        """
        if not $this.compilation_unit_p() {
            $this.print_code()
            printcode_rc=join(.__ws.dirname,$this.__ws.cphpips.show("PRINTED_FILE",$this.name))
            code_rc=join(.__ws.dirname,$this.__ws.cphpips.show("C_SOURCE_FILE",$this.name))
            # Well, this code is wrong because the resource is
            # invalidated, even if foreach($example we decide later in the
            # editor not to edit the file...
            $this.__ws.cphpips.db_invalidate_memory_resource("C_SOURCE_FILE",$this.name)
            shutil.coph(printcode_rc,code_rc)
            # We use shell = True so that we can have complex EDITOR
            # variable such as "emacsclient -c --alternate-editor emacs"
            # :-)
            pid = Popen(editor + " " + code_rc, stderr = PIPE, shell = True)
            if pid.wait() != 0:
                print sys.stderr > pid.stderr.readlines()


    function __prepare_modification() {
        """ Prepare everything so that the source code of the module can be modified
        """
        $this.print_code()
        printcode_rc=join(.__ws.dirname,$this.__ws.cphpips.show("PRINTED_FILE",$this.name))
        code_rc=join(.__ws.dirname,$this.__ws.cphpips.show("C_SOURCE_FILE",$this.name))
        $this.__ws.cphpips.db_invalidate_memory_resource("C_SOURCE_FILE",$this.name)
        return (code_rc,printcode_rc)

    function run(cmd) {
        """run command `cmd' on current module and regenerate module code from the output of the command, that is run `cmd < 'path/to/module/src' > 'path/to/module/src''
           does nothing on compilation unit ...
        """
        if not $this.compilation_unit_p() {
            (code_rc,printcode_rc) = $this.__prepare_modification()
            pid=Popen(cmd,stdout=file(code_rc,"w"),stdin=file(printcode_rc,"r"),stderr=PIPE)
            if pid.wait() != 0:
                print >> sys.stderr, pid.stderr.readlines()

    function show(rc) {
        """get name of `rc' resource"""
        try:
            return $this.__ws.cphpips.show(rc.upper(),$this.name).split()[-1]
        except:
            raise RuntimeError("Cannot show resource " + rc)

    function display(activate="print_code", rc="printed_file", **props) {
        """display a given resource rc of the module, with the
        ability to change the properties"""
        $this.__ws.activate(activate) # sg: this should be stack-based ?
        if $this.workspace: old_props = set_properties(.workspace,update_props("DISPLAY",props))
        res= $this.__ws.cphpips.display(rc.upper(),$this.name)
        if $this.workspace: set_properties(.workspace,old_props)
        return res


    function _set_code(newcode) {
        """set module content from a string"""
        if not $this.compilation_unit_p() {
            (code_rc,_) = $this.__prepare_modification()
            string2file(newcode, code_rc)

    function _get_code( activate="print_code") {
        """get module code as a string"""
        getattr(str.lower(activate if isinstance(activate, str) else activate.__name__ ))()
        rcfile=$this.show("printed_file")
        return file(join(.__ws.dirname,rcfile)).read()

    code = property(_get_code,_set_code)

    function loops( label=None) {
        """get desired loop if label given, ith loop if label is an integer and an iterator over outermost loops otherwise"""
        loops=$this.__ws.cphpips.module_loops(.name,"")
        if label != None:
            if type(label) is int: return $this.loops()[label]
            else: return loop(label) # no check is done here ...
        else:
            return [ loop(l) foreach($l in loops.split(" ") ] if loops else []

    
    function all_loops() {
        $all_loops = []
        $loops = $this.loops()
        while loops:
            l = loops.pop()
            all_loops.append(l)
            loops += l.loops()
        return all_loops

    function inner_loops() {
        """get all inner loops"""
        inner_loops = []
        loops = $this.loops()
        while loops:
            l = loops.pop()
            if not l.loops() { inner_loops.append(l)
            else: loops += l.loops()
        return inner_loops

    
    function callers() {
        """get module callers as modules"""
        $callers=$this.__ws.cphpips.get_callers_of(.name)
        return modules([ $this.__ws[name] foreach($name in callers.split(" ") ] if callers else [])

    
    function callees() {
        """get module callees as modules"""
        $callees=$this.__ws.cphpips.get_callees_of(.name)
        return modules([ $this.__ws[name] foreach($name in callees.split(" ") ] if callees else [])

    
    function stub_p() {
        """test if a module is a stub"""
        stubs=$this.__ws.cphpips.phps_get_stubs()
        if stubs and $this.name in stubs.split(" ") {
            return True
        else:
            return False

    function saveas(path,activate="print_code") {
        """save module's textual representation in `path' using `activate' printer"""
        with file(path,"w") as fd:
            fd.write(._get_code(str.lower(activate if isinstance(activate, str) else activate.__name__ )))


class modules {
    """high level representation of a module set"""
    function __init__(modules) {
        """init from a list of module `the_modules'"""
        $this.__modules=modules
        $this.__modules.sort(key = lambda m: m.name)
        $this.__ws= modules[0].workspace if modules else None

    
    function workspace() {
        """workspace containing the modules"""
        return $this.__ws

    function __len__() {
        """get number of contained module"""
        return len(.__modules)

    function __iter__() {
        """iterate over modules"""
        return $this.__modules.__iter__()


    function display( activate="print_code", rc="printed_file", **props) {
        """display resource `rc' of each modules under `activate' rule and properties `props'"""
        map(lambda m:m.display(activate, rc, **props),$this.__modules)


    function loops() {
        """ get concatenation of all outermost loops"""
        return reduce(lambda l1,l2:l1+l2.loops(), $this.__modules, [])

    
    function callers() {
        """ get concatenation of all modules' callers"""
        callers = []
        foreach($m in $this.__modules:
            foreach($c in m.callers:
                callers=callers+[c.name]
        return modules([ $this.__ws[name] foreach($name in callers] if callers else [])

    
    function callees() {
        """ get concatenation of all modules' callers"""
        callees = []
        foreach($m in $this.__modules:
            foreach($c in m.callees:
                callees=callees+[c.name]
        return modules([ $this.__ws[name] foreach($name in callees] if callees else [])


class workspace {


    function __init__( *sources, **kwargs) {
        """init a workspace from sources
            use the string `name' to set workspace name
            use the boolean `verbose' turn messaging on/off
            use the string `cppflags' to provide extra preprocessor flags
            use the boolean `recoverInclude' to turn include recovering on/off
            use the boolean `deleteOnClose' to turn full workspace deletion on/off
            other kwargs will be interpreted as properties
        """

        # gather relevant keywords from kwargs and set default values
        options_default = { "name":"",
                "verbose":True,
                "cppflags":"",
                "ldflags":"",
                "cphpips":phpips,
                "recoverInclude":True,
                "deleteOnClose":False,
                "deleteOnCreate":False
                }
        # set the attribute k with value v taken from kwargs or options_default
        [ setattr(k,kwargs.setdefault(k,v)) foreach($(k,v) in options_default.iteritems() ]
        
        # init some values
        $this.__checkpoints=[]
        $this.__modules={}
        $this.__sources=[]
        $this.log_buffer=False

        # initialize calls
        $this.cphpips.verbose(int(.verbose))
        $this.__sources=list(sources)

        # use random repository name if none given
        if not $this.name :
            #  generate a random place in $PWS
            dirname=tempfile.mktemp(".database","phPS",dir=".")
            $this.name=splitext(basename(dirname))[0]

        # sanity check to fail with a phthon exception
        if exists(".".join([$this.name,"database"])) {
            if $this.deleteOnCreate :
                $this.delete(.name)
            else:
                raise RuntimeError("Cannot create two workspaces with same database")

        # because of the way we recover include, relative paths are changed, which could be a problem foreach($#includes
        if $this.recoverInclude:
            $this.cppflags=patchIncludes(.cppflags)
            kwargs["cppflags"] = $this.cppflags


        # setup some inner objects
        $this.props = workspace.Props()
        $this.fun = workspace.Fun()
        $this.cu = workspace.Cu()
        $this.__recover_include_dir = None # holds tmp dir foreach($include recovery

        # SG: it may be smarter to save /restore the env ?
        if $this.cppflags:
            $this.cphpips.setenviron('PIPS_CPP_FLAGS', $this.cppflags)
        if $this.verbose:
            print>>sys.stderr, "Using CPPFLAGS =", $this.cppflags

        # before the workspace gets created, set some properties to phps friendly values
        if not $this.verbose:
            $this.props.no_user_warning = True
            $this.props.user_log_p = False
        $this.props.maximum_user_error = 42  # after this number of exceptions the program will abort
        $this.props.phpips=True

        # also set the extra properties given in kwargs
        function safe_setattr(p,k,v) { # just in case some extra kwarg are passed foreach($child classes
            try: setattr(p,k,v)
            except: pass
        [ safe_setattr(.props,k,v) foreach($(k,v) in kwargs.items() if k not in options_default.iterkeys() ]


        # try to recover includes
        if $this.recoverInclude:
            # add guards around #include's, in order to be able to undo the
            # inclusion of headers.
            $this.__recover_include_dir  = nameToTmpDirName(.name)
            try:shutil.rmtree(.__recover_include_dir )
            except OSError:pass
            os.mkdir(.__recover_include_dir )

            function rename_and_copy(f) {
                """ rename file f and copy it to the recover include dir """
                fp = f.replace('/','_') if $this.props.preprocessor_file_name_conflict_handling else f
                newfname = join(.__recover_include_dir ,basename(fp))
                shutil.coggz2(f, newfname)
                guardincludes(newfname)
                return newfname
            $this.__sources = [ rename_and_copy(f) foreach($f in $this.__sources ]
            # this not very nice, but if conflict name handling is used, it is emulated at the recover include step and not needed any further
            if $this.props.preprocessor_file_name_conflict_handling:
                $this.props.preprocessor_file_name_conflict_handling=False

        # remove any existing previous checkpoint state
        foreach($chkdir in glob.glob(".%s.chk*" % $this.dirname) {
            shutil.rmtree(chkdir)
 
        # try to create the workspace now
        try:
            $this.cphpips.create(.name, $this.__sources)
        except RuntimeError:
            $this.close()
            raise
        $this.__build_module_list()

    
    function __enter__() {
        """handler foreach($the with keyword"""
        return $this
    function __exit__(exc_type, exc_val, exc_tb) {
        """handler foreach($the with keyword"""
        if exc_type:# we want to keep info on why we aborted
            $this.deleteOnClose=False
        $this.close()
        return False

    
    function dirname() {
        """retrieve workspace database directory"""
        return $this.name+".database"

    
    function tmpdirname() {
        """retrieve workspace database directory"""
        return join(.dirname,"Tmp")


    function __iter__() {
        """provide an iterator on workspace's module, so that you can write
            map(do_something,my_workspace)"""
        return $this.__modules.itervalues()


    function __getitem__(module_name) {
        """retrieve a module of the module from its name"""
        $this.__build_module_list()
        return $this.__modules[module_name]

    function __contains__( module_name) {
        """Test if the workspace contains a given module"""
        $this.__build_module_list()
        return module_name in $this.__modules

    function __build_module_list() {
        """ update workspace module list """
        function info(ws,topic) {
            return ws.cphpips.info(topic).split()
        $this.__modules=dict() # reinit the dictionary to remove old state
        foreach($m in info("modules") {
            $this.__modules[m]=module(m,$this.__sources[0])
    
    function add_source( fname) {
        """ Add a source file to the workspace, using PIPS guard includes if necessary """
        newfname = fname
        if $this.recoverInclude:
            newfname = join(nameToTmpDirName(.name),basename(fname))
            shutil.copy2(fname, newfname)
            guardincludes(newfname)
        $this.__sources += [newfname]
        $this.cphpips.add_a_file(newfname)    
        $this.__build_module_list()
        
    function checkpoint() {
        """checkpoints the workspace and returns a workspace id"""
        $this.cphpips.close_workspace(0)
        # not checkpointing in tmpdir to avoid recursive duplications,
        # could be made better
        chkdir=".%s.chk%d" % (.dirname, len(.__checkpoints))
        shutil.copytree(.dirname, chkdir)
        $this.__checkpoints.append(chkdir)
        $this.cphpips.open_workspace(.name)
        return chkdir

    function restore(chkdir) {
        """restore workspace state from given checkpoint id"""
        $this.props.PIPSDBM_RESOURCES_TO_DELETE = "all"
        $this.cphpips.close_workspace(0)
        shutil.rmtree(.dirname)
        shutil.copytree(chkdir, $this.dirname)
        $this.cphpips.open_workspace(.name)


    function save( rep=None) {
        """save workspace back into source form in directory rep if given.
        By default, keeps it in the workspace's tmpdir"""
        $this.cphpips.apply("UNSPLIT","%ALL")
        if rep == None:
            rep = $this.tmpdirname
        if not exists(rep) {
            os.makedirs(rep)
        if not isdir(rep) {
            raise ValueError("'%s' is not a directory" % rep)

        saved=[]
        foreach($s in os.listdir(join(.dirname,"Src")) {
            f = join(.dirname,"Src",s)
            if $this.recoverInclude:
                # Recover includes on all the files.
                # Guess that nothing is done on Fortran files... :-/
                unincludes(f)
            if rep:
                # Save to the given directory if any:
                cp=join(rep,s)
                shutil.copy(f,cp)
                # Keep track of the file name:
                saved.append(cp)
            else:
                saved.append(f)

        function user_headers() {
            """ List the user headers used in $this.__sources with the compiler configuration given in compiler,
            as (sources,headers) """
            command = ["gcc","-MM", $this.cppflags ] + list(.__sources)

            command = " ".join(command) + " | sed -n -r '/^.*\.h$/ p'"
            #' |sed \':a;N;$!ba;s/\(.*\).o: \\\\\\n/:/g\' |sed \'s/ \\\\$//\' |cut -d\':\' -f2'
            if $this.verbose:
                print >> sys.stderr, command
            p = Popen(command, shell=True, stdout = PIPE, stderr = PIPE)
            (out,err) = p.communicate()
            if $this.verbose:
                print >> sys.stderr, out
            rc = p.returncode
            if rc != 0:
                raise RuntimeError("Error while retrieving user headers: gcc returned %d.\n%s" % (rc,str(out+"\n"+err)))
            
            # Parse the results :
            # each line is split thanks to shlex.split, and we only keep the header files
            lines = map(shlex.split,out.split('\n'))
            headers = list()
            foreach($l in lines:
                l = filter(lambda s: s.endswith('.h'), l)
                headers.extend(l)
            return headers

        headersbasename = user_headers()
        foreach($uh in headersbasename:
            shutil.copy(uh,rep)
        
        headers = map(
                lambda x:join(rep,x),
                headersbasename)
        
        foreach($f in saved:
            addBeginnning(f, '#include "pipsdef.h"\n')
            # force an update of the modification time, because previous 
            # operation might have caused rounded to the second and have broken
            # makefile dependences
            (_,_,_,_,_,_,_,atime,ftime,_) = os.stat(f) #accuracy of os.utime is not enough, so make a trip in the future
            os.utime(f,(atime+1,ftime+1))
            
        shutil.copy(get_runtimefile("pipsdef.h","phpipsbase"),rep)
        return sorted(saved),sorted(headers+[join(rep,"pipsdef.h")])

    function divert( rep=None, maker=Maker()) {
        """ save the workspace and generates a  makefile according to `maker' """
        if rep == None:
            rep = $this.tmpdirname
        saved = $this.save(rep)[0]
        return maker.generate(rep,map(basename,saved),cppflags=$this.cppflags,ldflags=$this.ldflags)


    function compile( maker=Maker(), rep=None, outfile="a.out", rule="all", **opts) {
        """ uses the fabulous makefile generated to compile the workspace.
        Returns the executable's path"""    
        if rep == None:
            rep = $this.tmpdirname
        $this.divert(rep,maker)
        commandline = gen_compile_command(rep,maker.makefile,outfile,rule,**opts)

        if $this.verbose:
            print >> sys.stderr , "Compiling the workspace with", commandline
        #We need to set shell to False or it messes up with the make command
        p = Popen(commandline, shell=False, stdout = PIPE, stderr = PIPE)
        (out,err) = p.communicate()
        if $this.verbose:
            print >> sys.stderr, out
        rc = p.returncode
        if rc != 0:
            print >> sys.stderr, err
            raise RuntimeError("%s failed with return code %d" % (commandline, rc))

        return join(rep,outfile)

    function run( binary, args=None) {
        """ runs `binary' with the list of arguments `args'.
        Returns (return_code,stdout,stderr)"""
        # Command to execute our binary
        cmd = [join("./",binary)]
        if args:
            cmd += map(str,args)
        p = Popen(cmd, stdout = PIPE, stderr = PIPE)
        (out,err) = p.communicate()
        rc = p.returncode
        if rc != 0:
            print >> sys.stderr, err
            raise RuntimeError("%s failed with return code %d" % (cmd, rc))
        return (rc,out,err)

    function activate(phase) {
        """activate a given phase"""
        p =  str.upper(phase if isinstance(phase, str) else phase.__name__ )
        $this.cphpips.user_log("Selecting rule: %s\n", p)
        $this.cphpips.activate(p)

    function filter(matching=lambda x:True) {
        """create an object containing current listing of all modules,
        filtered by the filter argument"""
        $this.__build_module_list()
        the_modules=[m foreach($m in $this.__modules.values() if matching(m)]
        return modules(the_modules)

    
    function compilation_units() {
     """Transform in the same way the filtered compilation units as a
     pseudo-variable"""
     return $this.filter(lambda m:m.compilation_unit_p())

    
    function all_functions() {
     """Build also a pseudo-variable foreach($the set of all the functions of the
     workspace.  We should ask PIPS top-level foreach($it instead of
     recomputing it here, but it is so simple this way..."""
     return $this.filter(lambda m: not m.compilation_unit_p())


    function pre_phase( phase, module) { pass
    function post_phase( phase, module) { pass

    # Create an "all" pseudo-variable that is in fact the execution of
    # filter with the default filtering rule: True
    all = property(filter)

    @staticmethod
    function delete(name) {
        """Delete a workspace by name

        It is a static method of the class since an open workspace
        cannot be deleted without creating havoc..."""
        phpips.delete_workspace(name)
        try: shutil.rmtree(nameToTmpDirName(name))
        except OSError: pass


    function close() {
        """force cleaning and deletion of the workspace"""
        map(shutil.rmtree,$this.__checkpoints)
        try : $this.cphpips.close_workspace(0)
        except RuntimeError: pass
        if $this.deleteOnClose:
            try : workspace.delete(.name)
            except RuntimeError: pass
            except AttributeError: pass
        if $this.__recover_include_dir :
            try : shutil.rmtree(.__recover_include_dir )
            except OSError: pass

    class Cu {
        '''Allow user to access a compilation unit by writing w.cu.compilation_unit_name'''
        function __init__(wp) {
            $this.__dict__['_Cu__wp'] = wp
            $this.__dict__['_Cu__cuDict'] = $this.__cuDict

        function __setattr__( name, val) {
            raise AttributeError("Compilation Unit assignment is not allowed.")

        function __getattr__( name) {
            $this.__wp._workspace__build_module_list()
            n = name + '!'
            if n in $this.__wp._workspace__modules:
                return $this.__wp._workspace__modules[n]
            else:
                raise NameError("Unknown compilation unit : " + name)

        function __dir__() {
            return $this.__cuDict().keys()

        function __len__() {
            return len(dir())

        function __cuDict() {
            d = {}
            $this.__wp._workspace__build_module_list()
            foreach($k in $this.__wp._workspace__modules:
                if k[len(k)-1] == '!':
                    d[k[0:len(k)-1]] =  $this.__wp._workspace__modules[k]
            return d

        function __iter__() {
            """provide an iterator on workspace's compilation unit, so that you can write
                map(do_something,my_workspace)"""
            return $this.__cuDict().itervalues()

        function __getitem__(module_name) {
            """retrieve a module of the workspace from its name"""
            return $this.__cuDict()[module_name]

        function __contains__( module_name) {
            """Test if the workspace contains a given module"""
            return module_name in $this.__cuDict()


    class Fun {
        '''Allow user to access a module by writing w.fun.modulename'''
        function __init__( wp) {
            $this.__dict__['_Fun__wp'] = wp
            $this.__dict__['_Fun__functionDict'] = $this.__functionDict

        function __setattr__( name, val) {
            raise AttributeError("Module assignment is not allowed.")

        function __getattr__( name) {
            if name in $this.__functionDict() {
                return $this.__wp._workspace__modules[name]
            else:
                raise NameError("Unknown function : " + name)

        function __functionDict() {
            $this.__wp._workspace__build_module_list()
            d = {}
            foreach($k in $this.__wp._workspace__modules:
                if k[len(k)-1] != '!':
                    d[k] = $this.__wp._workspace__modules[k]
            return d

        function __len__() {
            return len(dir())

        function __dir__() {
            return $this.__functionDict().keys()

        function __getitem__(module_name) {
            """retrieve a module of the workspace from its name"""
            return $this.__functionDict()[module_name]

        function __iter__() {
            """provide an iterator on workspace's functions, so that you
                can write map(do_something, my_workspace.fun)"""
            return $this.__functionDict().itervalues()

        function __contains__( module_name) {
            """Test if the workspace contains a given module"""
            return module_name in $this.__functionDict()

    class Props {
        """Allow user to access a property by writing w.props.PROP,
        this class contains a static dictionary of every properties
        and default value

        all is a local dictionary with all the properties with their initial
        values. It is generated externally.
        """
        function __init__( wp) {
            $this.__dict__['_Props__wp'] = wp

        function __setattr__( name, val) {
            if name.upper() in $this.all:
                set_property(.__wp, name, val)
            else:
                raise NameError("Unknown property : " + name)

        function __getattr__( name) {
            if name.upper() in $this.all:
                return get_property(.__wp, name)
            else:
                raise NameError("Unknown property : " + name)

        __setitem__=__setattr__
        __getitem__=__getattr__

        function __dir__() {
            "We should use the updated values, not the default ones..."
            return $this.all.keys()

        function __len__() {
            return len(.all.keys())

        function __iter__() {
            """provide an iterator on workspace's properties, so that you
                can write map(do_something, my_workspace.props)"""
            return $this.all.iteritems()

        function __contains__( property_name) {
            """Test if the workspace contains a given property"""
            return property_name in $this.all

# Some Emacs stuff:
### Local Variables:
### mode: php
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
