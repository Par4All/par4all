# coding=iso-8859-15
import pypips
import utils
import os
import sys
import tempfile
import shutil
import re
import time
import types
from copy import deepcopy
from string import split, upper, join
from subprocess import Popen, PIPE
import inspect

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
		loops=self._ws.cpypips.module_loops(self._module.name,self._label)
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
		printcode_rc=os.path.join(self._ws.directory(),self._ws.cpypips.show("PRINTED_FILE",self.name))
		code_rc=os.path.join(self._ws.directory(),self._ws.cpypips.show("C_SOURCE_FILE",self.name))
		thecmd=cmd+[printcode_rc]
		self._ws.cpypips.db_invalidate_memory_resource("C_SOURCE_FILE",self._name)
		pid=Popen(thecmd,stdout=file(code_rc,"w"),stderr=PIPE)
		if pid.wait() != 0:
			print sys.stderr > pid.stderr.readlines()

	def show(self,rc):
		"""returns the name of resource rc"""
		return split(self._ws.cpypips.show(upper(rc),self._name))[-1]

	def apply(self, phase, *args, **kwargs):
		self._ws.cpypips.apply(upper(phase),self._name)

	def display(self,rc="printed_file",With="PRINT_CODE", **props):
		"""display a given resource rc of the module, with the
		ability to change the properties"""
		self._ws.activate(With)
		self._ws.set_properties(props)
		return self._ws.cpypips.display(upper(rc),self._name)

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
			loops=self._ws.cpypips.module_loops(self.name,"")
			if not loops:
				return []
			return map(lambda l:loop(self,l),loops.split(" "))
	
	def callers(self):
		callers=self._ws.cpypips.get_callers_of(self.name)
		if not callers:
			return []
		return callers.split(" ")

	def callees(self):
		callees=self._ws.cpypips.get_callees_of(self.name)
		if not callees:
			return []
		return callees.split(" ")

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
		if self._modules: self._ws.cpypips.capply(upper(phase),map(lambda m:m.name,self._modules))

### modules_methods /!\ do not touch this line /!\

class workspace(object):
	"""Top level element of the pyps hierarchy, it represents a set of source
	   files and provides methods to manipulate them.

	    Of note is the `parents' attribute, which allow to define advices
		before, around and after the initial functions.

		In order to use this, create a workspace, for instance:

		# File foo.py
		class workspace:
			# No need to inherit
			def __init__(self, ws, sources, **kwargs):
				# Given the same arguments as pyps.workspace, plus the workspace
				# itself as the first argument
				pass # nothing to do, maybe `self.ws = ws' to save the `real'
					 # workspace if necessary.

			def pre_bar(self, arg...):
				# Same arguments as pyps.workspace.bar()
				print "foo.pre_bar: running just before pyps.workspace.bar()"

		The users will need to `import foo', and create their workspace as `ws =
		pyps.workspace(sources, parents = [foo.workspace])'.

		With a `pre_' suffix, the function will be called before the function of
		the same name; with a `post_', it will be run after. Without a prefix,
		it will entirely replace the initial function. In other words, defining
		pre_bar is similar to saying in emacs:

		(defadvice foo-pre-bar (before bar (arg...) (progn ...)))
		(ad-activate foo-pre-bar)

		Defining foo.workspace.baz() when pyps.workspace.baz() already exists
		will *replace* pyps.workspace.baz() (of course, foo's version of baz()
		may call the initial one, think (defadvice foo-baz (around baz ...) ...)
		XXX: See comments around foundMain in __getattribute__().

		If the workspace is created as: `ws = pyps.workspace(..., parents =
		[foo, bar])', a call to ws.quux will call:

		foo.pre_quux
		bar.pre_quux
		foo.quux
			at foo.quux' discretion, bar.quux
				at bar.quux's discretion, pyps.workspace.quux
		bar.post_quux
		foo.post_quux

		This is not an error if any (or even all) intermediate functions don't
		exist.
		"""

	def __init__(self,sources2,name="",activates=[],verboseon=True,cppflags='', parents=[], cpypips = None, recoverInclude=True):
		kwargs = {'name':name, 'activates':activates, 'verboseon':verboseon, 'cppflags':cppflags, 'parents':parents, 'recoverInclude':parents }

		if cpypips == None:
			cpypips = pypips
		self.cpypips = cpypips

		self.recoverInclude=recoverInclude

		#In case the subworkspaces need to add files, the variable passed in parameter will only
		#be modified here and not in the scope of the caller
		sources2 = deepcopy(sources2)
		# Do this first as other workspaces may want to modify sources
		# (sac.workspace does).
		self.iparents = []
		for p in parents:
			pws = p(self, sources2, **kwargs)
			self.iparents.append(pws)

		"""init a workspace from a list of sources"""
		self._modules = {}
		self.props = workspace.props(self)
		self.fun = workspace.fun(self)
		self.cu = workspace.cu(self)

		if name == "":
			name=os.path.basename(tempfile.mkdtemp("","PYPS"))
		# SG: it may be smarter to save /restore the env ?
		if cppflags != "":
			os.environ['PIPS_CPP_FLAGS']=cppflags

		def helper(x,y):
			return x+y if isinstance(y,list) else x +[y]
		sources2 = reduce(helper, sources2, [])
		sources = []

		if recoverInclude:
			# add guards around #include's, in order to be able to undo the
			# inclusion of headers.
			self.tmpDirName = utils.nameToTmpDirName(name)
			os.mkdir(self.tmpDirName)

			for f in sources2:
				newfname = os.path.join(self.tmpDirName,os.path.basename(f))
				shutil.copy2(f, newfname)
				sources += [newfname]
				utils.guardincludes(newfname)
		else:
			sources=sources2

		self._sources = sources

		try:
			cpypips.create(name, self._sources)
		except RuntimeError:
			cpypips.quit()
			cpypips.delete_workspace(name)
			raise

		if not verboseon:
			self.props.NO_USER_WARNING = True
			self.props.USER_LOG_P = False
		self.props.MAXIMUM_USER_ERROR = 42  # after this number of exceptions the programm will abort
		map(lambda x:cpypips.activate(x),activates)
		self._build_module_list()
		self._name=self.info("workspace")[0]
		
		"""Call all the functions 'post_init' of the given parents"""
		for pws in reversed(self.iparents):
			try:
				pws.post_init(sources, **kwargs)
			except AttributeError:
				pass

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
		return split(self.cpypips.info(topic))

	def bckdir(self):
		return os.path.dirname(self.directory()) + ".bck"

	def directory(self):
		"""retrieve workspace database directory"""
		return self._name+".database/"

	def checkpoint(self):
		self.cpypips.checkpoint()
		if os.path.exists(self.bckdir()):
			shutil.rmtree(self.bckdir())
		shutil.copytree(self.directory(), self.bckdir())

	def restore(self):
		self.props.PIPSDBM_RESOURCES_TO_DELETE = "all"
		self.cpypips.quit()
		shutil.rmtree(self.directory())
		shutil.copytree(self.bckdir(), self.directory())
		self.cpypips.restore_open_workspace(self.name)

	def get_property(self, name):
		name = upper(name)
		"""return property value"""
		t = type(self.props.all[name])

		if t == str:     return self.cpypips.get_string_property(name)
		elif t == int:   return self.cpypips.get_int_property(name)
		elif t == bool : return self.cpypips.get_bool_property(name)
		else : 
			raise TypeError( 'Property type ' + str(t) + ' isn\'t supported')

	def get_properties(self, props):
		"""return a list of values of props list"""
		res = []
		for prop in props.iteritems():
			res.append(get_property(self, prop))
		return res

	def _set_property(self, prop,value):
		"""change property value and return the old one"""
		prop = upper(prop)
		old = self.get_property(prop)
		if value == None:
			return old
		if type(value) is bool:
			val=upper(str(value))
		elif type(value) is str:
			def stringify(s): return '"'+s+'"'
			val=stringify(value)
		else:
			val=str(value)
		self.cpypips.set_property(upper(prop),val)
		return old

	def set_properties(self,props):
		"""set properties based on the dictionnary props and returns a dictionnary containing the old state"""
		old = dict()
		for prop,value in props.iteritems():
			old[prop] = self._set_property(prop, value)
		return old

	def set_property(self, **props):
		"""set properties and return a dictionnary containing the old state"""
		return self.set_properties(props)

	def save(self,indir="",with_prefix=""):
		"""save workspace back into source either in directory indir or with the prefix with_prefix"""
		self.cpypips.apply("UNSPLIT","%ALL")
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
		if self.recoverInclude:
			for f in saved:
				utils.unincludes(f)
		return saved

	def compile(self,CC="gcc",CFLAGS="-O2 -g", LDFLAGS="", link=True, outdir=".", outfile="",extrafiles=[]):
		"""try to compile current workspace, some extrafiles can be given with extrafiles list"""
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
		commandline = " ".join(command)
		#print "running", commandline
		ret = os.system(commandline)
		if ret:
			if not link: map(os.remove,otmpfiles)
			raise RuntimeError("`%s' failed with return code %d" % (commandline, ret >> 8))
		return outfile
	
	# allows subclasses to tamper with the files before compiling
	def goingToRunWith(self, files, outdir):
		for f in files:
			utils.addMAX0(f)

	def activate(self,phase):
		"""activate a given phase"""
		if isinstance(phase, str):
			p = upper(phase)
		else:
			p = upper(phase.__name__)
		self.cpypips.user_log("Selecting rule: %s\n", p)
		self.cpypips.activate(p)

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
		try: shutil.rmtree(utils.nameToTmpDirName(name))
		except OSError: pass


	def close(self):
		"""force cleaning and deletion of the workspace"""
		try : self.cpypips.quit()
		except RuntimeError: pass
		try : workspace.delete(self._name)
		except RuntimeError: pass
		self.hasBeenClosed = True

	def __del__(self):
		if(not hasattr(self, "hasBeenClosed")):
			close()

	def _build_module_list(self):
		for m in self.info("modules"):
			self._modules[m]=module(self,m,self._sources[0])
	
	def __getattribute__(self, name):
		"""Method overrinding the normal attribute access."""

		def get(name):
			# can't use self.name !
			return object.__getattribute__(self, name)
		def search(instance, iname):
			try:
				return instance.__class__.__dict__[iname]
			except KeyError:
				return None

		# Get rid of attribute (non-method) access
		try:
			a = get(name)
			if not callable(a) or inspect.isclass(a):
				return a
		except AttributeError:
			pass
			
		#Functions with '__' before can't be overloaded (for example __dir__), but they can still have
		# pre_ and post_ hooks. 
		if name[0:2] == '__':
			overloadable = False
		else:
			overloadable = True
		

		# Build a new function
		pre_hooks = []
		method = lambda *args, **kwargs: None
		post_hooks = []

		# List the pre_name hooks
		for p in get("iparents"):
			pre_hook = search(p, "pre_" + name)
			if callable(pre_hook):
				pre_hooks.append((p, pre_hook))

		# Find the correct "main" function.

		# First, try the redefinition from one of the parents, and if all fail,
		# try the initial one. The return value is that of the method really
		# called.

		# XXX: I may want to do the following in one of the parent class:
		# class foo_workspace:
		# 	def compile(self, **args):
		#		"""a method that compile then launch gdb"""
		#		args["CFLAGS"] += "-g"
		#		FIXME: call pyps.workspace.compile with **args
		#		os.system("gdb ....")

		# I don't know how to do the line marked with FIXME. Ideally, if there
		# are several workspaces defining compile(), they would stack correctly.
		# OTOH, I haven't found (yet?) a compelling use case...
		foundMain = False
		if overloadable:
			for p in get("iparents"):
				m = search(p, name)
				if callable(m):
					method = types.MethodType(m, p, type(p))
					foundMain = True
					break
		if not foundMain:
			try:
				method = get(name)
				foundMain = True
			except AttributeError:
				pass

		# List the post_name hooks
		for p in reversed(get("iparents")):
			post_hook = search(p, "post_" + name)
			if callable(post_hook):
				post_hooks.append((p, post_hook))

		# If we didn't fund anything, raise an error
		if (not foundMain) and pre_hooks == [] and post_hooks == []:
			raise AttributeError

		def ret(self, *args, **kwargs):
			for p, f in pre_hooks:
				f(p, *args, **kwargs)
			val = method(*args, **kwargs)
			for p, f in post_hooks:
				f(p, *args, **kwargs)
			return val
		ret.__doc__ = method.__doc__
		return  types.MethodType(ret, self, type(self))

	def __dir__(self):
		l = self.__class__.__dict__.keys()
		for p in self.iparents:
			l += dir(p)
		return l

	class cu(object):
		'''Allow user to access a module by writing w.fun.modulename'''
		def __init__(self,wp):
			self.__dict__['_wp'] = wp

		def __setattr__(self, name, val):
			raise AttributeError("Module assignement is not allowed.")

		def __getattr__(self, name):
			n = name + '!'
			if n in self._wp._modules:
				return self._wp._modules[n]
			else:
				raise NameError("Unknow compilation unit : " + name)

		def __dir__(self):
			return self._cuDict().keys()

		def _cuDict(self):
			d = {}
			for k in self._wp._modules:
				if k[len(k)-1] == '!':
					d[k[0:len(k)-1]] =  self._wp._modules[k]
			return d

		def __iter__(self):
			"""provide an iterator on workspace's module, so that you can write
				map(do_something,my_workspace)"""
			return self._cuDict().itervalues()
	
		def __pyropsFakeIter__(self):
			"""provide an iterator on workspace's module, so that you can write
				map(do_something,my_workspace)"""
			return self._cuDict().values()
	
		def __getitem__(self,module_name):
			"""retrieve a module of the module from its name"""
			return self._cuDict()[module_name]
	
	
		def __setitem__(self,i):
			"""change a module of the module from its name"""
			return self._cuDict()[i]
	
	
		def __contains__(self, module_name):
			"""Test if the workspace contains a given module"""
			return module_name in self._cuDict()

	class fun(object):
		'''Allow user to access a module by writing w.fun.modulename'''
		def __init__(self,wp):
			self.__dict__['_wp'] = wp

		def __setattr__(self, name, val):
			raise AttributeError("Module assignement is not allowed.")

		def __getattr__(self, name):
			if name in self._functionDict():
				return self._wp._modules[name]
			else:
				raise NameError("Unknow function : " + name)

		def _functionDict(self):
			d = {}
			for k in self._wp._modules:
				if k[len(k)-1] != '!':
					d[k] = self._wp._modules[k] 
			return d

		def __dir__(self):
			return self._functionDict().keys()

		def __iter__(self):
			"""provide an iterator on workspace's module, so that you can write
				map(do_something,my_workspace)"""
			return self._functionDict().itervalues()
	
		def __pyropsFakeIter__(self):
			"""provide an iterator on workspace's module, so that you can write
				map(do_something,my_workspace)"""
			return self._functionDict().values()
	
		def __getitem__(self,module_name):
			"""retrieve a module of the module from its name"""
			return self._functionDict()[module_name]
	
	
		def __setitem__(self,i):
			"""change a module of the module from its name"""
			return self._functionDict()[i]
	
	
		def __contains__(self, module_name):
			"""Test if the workspace contains a given module"""
			return module_name in self._functionDict()

	class props(object):
		"""Allow user to access a property by writing w.props.PROP,
		this class contains a static dictionnary of every properties
		and default value"""
		def __init__(self,wp):
			self.__dict__['wp'] = wp

		def __setattr__(self, name, val):
			if name in self.all:
				self.wp._set_property(name,val)
			else:
				raise NameError("Unknow property : " + name)

		def __getattr__(self, name):
			if name in self.all:
				return self.wp.get_property(name)
			else:
				raise NameError("Unknow property : " + name)

		def __dir__(self):
			return self.all.keys()

		def __iter__(self):
			"""provide an iterator on workspace's module, so that you can write
				map(do_something,my_workspace)"""
			return self._modules.itervalues()
	
		def __pyropsFakeIter__(self):
			"""provide an iterator on workspace's module, so that you can write
				map(do_something,my_workspace)"""
			return self._modules.values()
	
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

		@staticmethod
		def update_props(passe,props):
			"""Change a property dictionnary by appending the pass name to the property when needed """
			for name,val in props.iteritems():
				if upper(name) not in workspace.props.all:
					del props[name]
					props[upper(passe+"_"+name)]=val
					#print "warning, changing ", name, "into", passe+"_"+name
			return props
			

### props_methods /!\ do not touch this line /!\

# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### python-indent: 4
### End:
