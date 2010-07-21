# coding=iso-8859-15
import pypips
import os
import tempfile
import shutil
import re
import time
from string import split, upper, join

pypips.atinit()

class loop:
	"""a loop represent a do-loop of a module"""

	def __init__(self,module,label):
		"""[[internal]] bind a loop to its module"""
		self._module=module
		self._label=label
		self._ws=module._ws

	@property
	def label(self): return self._label
	
	@property
	def module(self): return self._module

	def display(self): self._module.display()

	def loops(self):
		self._module.flag_loops()
		loops=pypips.module_loops(self._module.name,self._label)
		if not loops:
			return []
		return map(lambda l:loop(self._module,l),str.split(loops," "))



### loop_methods /!\ do not touch this line /!\


class module:
	"""a module represents a function or a procedure, it is the basic
	element of PyPS you can select modules from the workspace
	and apply transformations to them"""

	def __init__(self,ws,name,source=""):
		"""[[internal]] bind a module to its workspace"""
		self._name=name
		self._source=source
		self._ws=ws

	@property
	def name(self): return self._name

	def run(self,cmd):
		"""runs command `cmd' on current module and regenerate module code from the output of the command, that is run `cmd 'path/to/module/src' > 'path/to/module/src''"""
		self.print_code()
		printcode_rc=os.path.join(self._ws.directory(),pypips.show("PRINTED_FILE",self.name))
		code_rc=os.path.join(self._ws.directory(),pypips.show("C_SOURCE_FILE",self.name))
		thecmd=cmd+[printcode_rc]
		pypips.db_invalidate_memory_resource("C_SOURCE_FILE",self._name)
		pid=Popen(thecmd,stdout=file(code_rc,"w"),stderr=PIPE)
		if pid.wait() != 0:
			print sys.stderr > pid.stderr.readlines()

	def show(self,rc):
		"""returns the name of resource rc"""
		return split(pypips.show(upper(rc),self._name))[-1]

	def apply(self,phase):
		"""apply transformation phase"""
		pypips.apply(upper(phase),self._name)

	def display(self,rc="printed_file",With="PRINT_CODE", **props):
		"""display a given resource rc of the module, with the
		ability to change the properties"""
		self._ws.activate(With)
		self._ws._set_property(self._update_props("display", props))
		return pypips.display(upper(rc),self._name)

	def code(self):
		"""return module code as a string"""
		self.apply("print_code")
		rcfile=self.show("printed_file")
		return file(self._ws.directory()+rcfile).readlines()

	def loops(self, label=""):
		"""return desired loop if label given, an iterator over loops otherwise"""
		if label: return loop(self,label)
		else:
			self.flag_loops()
			loops=pypips.module_loops(self.name,"")
			if not loops:
				return []
			return map(lambda l:loop(self,l),loops.split(" "))
	
	def callers(self):
		callers=pypips.get_callers_of(self.name)
		if not callers:
			return []
		return callers.split(" ")

	def _update_props(self,passe,props):
		"""[[internal]] change a property dictionnary by appending the pass name to the property when needed """
		for name,val in props.iteritems():
			if upper(name) not in self._all_properties:
				del props[upper(name)]
				props[upper(passe+"_"+name)]=val
				#print "warning, changing ", name, "into", passe+"_"+name
		return props
	def saveas(self,path):
		fd=file(path,"w")
		for line in self.code():
			fd.write(line)
		fd.close()

### module_methods /!\ do not touch this line /!\

class modules:
	"""high level representation of a module set,
	its only purpose is to dispatch method calls on contained modules"""
	def __init__(self,modules):
		self._modules=modules
		self._ws= modules[0]._ws if modules else None


	def display(self,rc="printed_file", With="PRINT_CODE", **props):
		"""display all modules by default with the code and some
		eventual property setting"""
		map(lambda m:m.display(rc, With, **props),self._modules)


	def loops(self):
		""" return a list of all program loops"""
		return reduce(lambda l1,l2:l1+l2.loops(), self._modules, [])

	def capply(self,phase):
		""" concurrently apply a phase to all contained modules"""
		pypips.capply(upper(phase),map(lambda m:m.name,self._modules))

### modules_methods /!\ do not touch this line /!\

class workspace(object):
	"""top level element of the pyps hierarchy,
		it represents a set of source files and provides methods
		to manipulate them"""

	def __init__(self,sources2,name="",activates=[],verboseon=True,cppflags=''):
		"""init a workspace from a list of sources"""
		self._modules = {}
		if name == "":
			name=os.path.basename(tempfile.mkdtemp("","PYPS"))
		# SG: it may be smarter to save /restore the env ?
		if cppflags != "":
			os.environ['PIPS_CPP_FLAGS']=cppflags
		def helper(x,y):
			return x+y if isinstance(y,list) else x +[y]
		self._sources=reduce(helper,sources2,[])
		pypips.create(name, self._sources)
		if not verboseon:
			self.set_property(NO_USER_WARNING=True)
			self.set_property(USER_LOG_P=False)
		self.set_property(MAXIMUM_USER_ERROR=42)  # after this number of exceptions the programm will abort
		map(lambda x:pypips.activate(x),activates)
		self._build_module_list()
		self._name=self.info("workspace")[0]

	@property
	def name(self):return self._name

	def __iter__(self):
		"""provide an iterator on workspace's module, so that you can write
			map(do_something,my_workspace)"""
		return self._modules.itervalues()


	def __getitem__(self,module_name):
		"""retrieve a module of the module from its name"""
		self._build_module_list()
		return self._modules[module_name]


	def __setitem__(self,i):
		"""change a module of the module from its name"""
		return self._modules[i]


	def __contains__(self, module_name):
		"""Test if the workspace contains a given module"""
		self._build_module_list()
		return module_name in self._modules


	def info(self,topic):
		return split(pypips.info(topic))

	def directory(self):
		"""retrieve workspace database directory"""
		return self._name+".database/"

	def _set_property(self,props):
		"""[internal] set properties based on the dictionnary props"""
		for prop,value in props.iteritems():
			if type(value) is bool:
				val=upper(str(value))
			elif type(value) is str:
				def stringify(s): return '"'+s+'"'
				val=stringify(value)
			else:
				val=str(value)
			pypips.set_property(upper(prop),val)

	def set_property(self,**props):
		"""set multiple properties at once"""
		self._set_property(props)

	def save(self,indir="",with_prefix=""):
		"""save workspace back into source either in directory indir or with the prefix with_prefix"""
		pypips.apply("UNSPLIT","%ALL")
		saved=[]
		if indir:
			if not os.path.exists(indir):
				os.makedirs(indir)
			if not os.path.isdir(indir): raise ValueError("'" + indir + "' is not a directory")
			for s in os.listdir(self.directory()+"Src"):
				cp=os.path.join(indir,s)
				shutil.copy(os.path.join(self.directory(),"Src",s),cp)
				saved+=[cp]
		else:
			for s in os.listdir(self.directory()+"Src"):
				cp=with_prefix+s
				shutil.copy(os.path.join(self.directory(),"Src",s),cp)
				saved+=[cp]
		return saved

	def compile(self,CC="gcc",CFLAGS="-O2 -g", LDFLAGS="", link=True, outdir=".", outfile="",extrafiles=[]):
		"""try to compile current workspace, some extrafiles can be given with extrafiles list"""
<<<<<<< HEAD
=======
		if not os.path.isdir(outdir): raise ValueError("'" + outdir + "' is not a directory")
>>>>>>> Made workspace inherit from object and added functions.
		otmpfiles=self.save(indir=outdir)+extrafiles
		command=[CC,CFLAGS]
		if link:
			if not outfile:
				outfile=self._name
			self.goingToRunWith(otmpfiles, outdir)
			command+=otmpfiles
			command+=[LDFLAGS]
			command+=["-o", outfile]
		else:
			self.goingToRunWith(otmpfiles, outdir)
			command+=["-c"]
			command+=otmpfiles
		#print "running", " ".join(command)
		if os.system(" ".join(command)):
			if not link: map(os.remove,otmpfiles)
		return outfile
	
	# allows subclasses to tamper with the files before compiling
	def goingToRunWith(self, files, outdir):
		pass

	def activate(self,phase):
		"""activate a given phase"""
		pypips.activate(upper(phase))

	def filter(self,matching=lambda x:True):
		"""create an object containing current listing of all modules,
		filtered by the filter argument"""
		self._build_module_list()
		#print self._modules.values()
		the_modules=[m for m in self._modules.values() if matching(m)]
		return modules(the_modules)


	# Create an "all" pseudo-variable that is in fact the execution of
	# filter with the default filtering rule: True
	all = property(filter)


	# A regex matching compilation unit names ending with a "!":
	re_compilation_units = re.compile("^.*!$")

	# Get the list of compilation units:
	def filter_compilation_units(self):
		return self.filter(lambda m: workspace.re_compilation_units.match(m.name))

	# Transform it in a pseudo-variable:
	compilation_units = property(filter_compilation_units)


	# A real function is a module that is not a compilation unit:
	def filter_all_functions(self):
		return self.filter(lambda m: not workspace.re_compilation_units.match(m.name))

	# Transform it in a pseudo-variable.  We should ask PIPS top-level
	# for it instead of recomputing it here, but it is so simple this
	# way...
	all_functions = property(filter_all_functions)

	@staticmethod
	def delete(name):
		"""Delete a workspace by name

		It is a static method of the class since an open workspace
		cannot be deleted without creating havoc..."""
		pypips.delete_workspace(name)


	def close(self):
		"""force cleaning and deletion of the workspace"""
		try :
			pypips.quit()
			workspace.delete(self._name)
		except RuntimeError:
			pass


	def _build_module_list(self):
		for m in self.info("modules"):
			self._modules[m]=module(self,m,self._sources[0])


# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
