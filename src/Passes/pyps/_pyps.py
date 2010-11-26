# coding=iso-8859-15
from __future__ import with_statement # to cope with python2.5
import pypips
import pypsutils
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
	"""do-loop from a module"""

	def __init__(self,module,label):
		"""[[internal]] bind a loop to its module"""
		self._module=module
		self._label=label
		self._ws=module._ws

	@property
	def label(self):
		"""loop label, as seen in the source code"""
		return self._label

	@property
	def module(self):
		"""module containing the loop"""
		return self._module

	def display(self):
		"""display loop module"""
		self._module.display()

	def loops(self):
		"""get outer loops of this loop"""
		loops=self._ws.cpypips.module_loops(self._module.name,self._label)
		return [ loop(self._module,l) for l in str.split(loops," ") ] if loops else []



### loop_methods /!\ do not touch this line /!\


class module:
	"""A source code function"""

	def __init__(self,ws,name,source=""):
		"""[[internal]] bind a module to its workspace"""
		self._name=name
		self._source=source
		self._ws=ws

	@property
	def cu(self):
		"""compilation unit"""
		return self._ws.cpypips.compilation_unit_of_module(self._name)[:-1]

	@property
	def name(self):
		"""module name"""
		return self._name

	def edit(self,editor=os.getenv("EDITOR","vi")):
		"""edits module using given editor
		   does nothing on compilation units ...
		"""
		if not pypsutils.re_compilation_units.match(self.name):
			self.print_code()
			printcode_rc=os.path.join(self._ws.dirname(),self._ws.cpypips.show("PRINTED_FILE",self.name))
			code_rc=os.path.join(self._ws.dirname(),self._ws.cpypips.show("C_SOURCE_FILE",self.name))
			self._ws.cpypips.db_invalidate_memory_resource("C_SOURCE_FILE",self._name)
			shutil.copy(printcode_rc,code_rc)
			pid=Popen([editor,code_rc],stderr=PIPE)
			if pid.wait() != 0:
				print sys.stderr > pid.stderr.readlines()

	def run(self,cmd):
		"""run command `cmd' on current module and regenerate module code from the output of the command, that is run `cmd < 'path/to/module/src' > 'path/to/module/src''
		   does nothing on compilation unit ...
		"""
		if not pypsutils.re_compilation_units.match(self.name):
			self.print_code()
			printcode_rc=os.path.join(self._ws.dirname(),self._ws.cpypips.show("PRINTED_FILE",self.name))
			code_rc=os.path.join(self._ws.dirname(),self._ws.cpypips.show("C_SOURCE_FILE",self.name))
			self._ws.cpypips.db_invalidate_memory_resource("C_SOURCE_FILE",self._name)
			pid=Popen(cmd,stdout=file(code_rc,"w"),stdin=file(printcode_rc,"r"),stderr=PIPE)
			if pid.wait() != 0:
				print sys.stderr > pid.stderr.readlines()

	def show(self,rc):
		"""get name of `rc' resource"""
		return split(self._ws.cpypips.show(upper(rc),self._name))[-1]

	def display(self,rc="printed_file",activate="print_code", **props):
		"""display a given resource rc of the module, with the
		ability to change the properties"""
		self._ws.activate(activate)
		pypsutils.set_properties(self._ws,props)
		return self._ws.cpypips.display(upper(rc),self._name)

	def code(self):
		"""get module code as a string list"""
		self._ws.cpypips.apply("PRINT_CODE",self._name)
		rcfile=self.show("printed_file")
		return file(self._ws.dirname()+rcfile).readlines()

	def loops(self, label=""):
		"""get desired loop if label given, an iterator over outermost loops otherwise"""
		loops=self._ws.cpypips.module_loops(self.name,"")
		if label:
			if label in loops: return loop(self,label)
			else: raise Exception("Loop label invalid")
		else:
			return [ loop(self,l) for l in loops.split(" ") ] if loops else []

	def inner_loops(self):
		"""get all inner loops"""
		inner_loops = []
		loops = self.loops()
		while loops:
			l = loops.pop()
			if not l.loops(): inner_loops.append(l)
			else: loops += l.loops()
		return inner_loops

	@property
	def callers(self):
		"""get module callers as modules"""
		callers=self._ws.cpypips.get_callers_of(self.name)
		return modules([ self._ws[name] for name in callers.split(" ") ] if callers else [])

	@property
	def callees(self):
		"""get module callees as modules"""
		callees=self._ws.cpypips.get_callees_of(self.name)
		return modules([ self._ws[name] for name in callees.split(" ") ] if callees else [])

	def _update_props(self,passe,props):
		"""[[internal]] change a property dictionary by appending the pass name to the property when needed """
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
	"""high level representation of a module set"""
	def __init__(self,modules):
		"""initi from a list of module`modules'"""
		self._modules=modules
		self._ws= modules[0]._ws if modules else None

	def __len__(self):
		"""get number of contained module"""
		return len(self._modules)

	def __iter__(self):
		"""iterate over modules"""
		return self._modules.__iter__()


	def display(self,rc="printed_file", activate="print_code", **props):
		"""display resource `rc' of each modules under `activate' rule and properties `props'"""
		map(lambda m:m.display(rc, activate, **props),self._modules)


	def loops(self):
		""" get concatenation of all outermost loops"""
		return reduce(lambda l1,l2:l1+l2.loops(), self._modules, [])

### modules_methods /!\ do not touch this line /!\

class workspace(object):

	"""Top level element of the pyps hierarchy, it represents a set of source
	   files and provides methods to manipulate them.
		"""

	def __init__(self, *sources, **kwargs):
		"""init a workspace from sources
			use the string `name' to set workspace name
			use the boolean `verbose' turn messaging on/off
			use the string `cppflags' to provide extra preprocessor flags
			use the list `parents' to customize the workspace
			use the boolean `recoverInclude' to turn include recoverning on/off
			use the boolean `deleteOnClose' to turn full workspace deletion on/off
		"""

		name           = kwargs.setdefault("name",           "")
		verbose        = kwargs.setdefault("verbose",        True)
		cppflags       = kwargs.setdefault("cppflags",       "")
		parents        = kwargs.setdefault("parents",        [])
		cpypips	       = kwargs.setdefault("cpypips",        pypips)
		recoverInclude = kwargs.setdefault("recoverInclude", True)
		deleteOnClose  = kwargs.setdefault("deleteOnClose",  False)


		self.cpypips = cpypips
		self.recoverInclude = recoverInclude
		self.verbose = verbose
		self.cpypips.verbose(int(verbose))
		self.deleteOnClose=deleteOnClose
		self.checkpoints=[]
		self._sources=[]

		if not name :
			#  generate a random place in $PWS
			dirname=tempfile.mktemp(".database","PYPS",dir=".")
			name=os.path.splitext(os.path.basename(dirname))[0]

		if os.path.exists(".".join([name,"database"])):
			raise RuntimeError("Cannot create two workspaces with same database")

		# because of the way we recover include, relative paths are changed, which could be a proplem for #includes
		if recoverInclude: cppflags=pypsutils.patchIncludes(cppflags)


		# In case the subworkspaces need to add files, the variable passed in
		# parameter will only be modified here and not in the scope of the caller
		sources2 = deepcopy(sources)
		# Do this first as other workspaces may want to modify sources
		# (sac.workspace does).
		self.iparents = []
		for p in parents:
			pws = p(self, sources2, **kwargs)
			self.iparents.append(pws)

		self._modules = {}
		self.props = workspace.props(self)
		self.fun = workspace.fun(self)
		self.cu = workspace.cu(self)
		self.tmpDirName= None # holds tmp dir for include recovery

		# SG: it may be smarter to save /restore the env ?
		if cppflags != "":
			self.cpypips.setenviron('PIPS_CPP_FLAGS', cppflags)
		self.cppflags = cppflags
		if self.verbose:
			print>>sys.stderr, "Using CPPFLAGS =", self.cppflags

		def concatenate_list(x, y):
			"Concatenate as list even if the second one is a simple element"
			return x + y if isinstance(y,list) else x + [y]
		sources2 = reduce(concatenate_list, sources2, [])
		sources = []

		if recoverInclude:
			# add guards around #include's, in order to be able to undo the
			# inclusion of headers.
			self.tmpDirName = pypsutils.nameToTmpDirName(name)
			try:shutil.rmtree(self.tmpDirName)
			except OSError:pass
			os.mkdir(self.tmpDirName)

			for f in sources2:
				newfname = os.path.join(self.tmpDirName,os.path.basename(f))
				shutil.copy2(f, newfname)
				sources += [newfname]
				pypsutils.guardincludes(newfname)
		else:
			sources=sources2

		self._sources += sources

		try:
			cpypips.create(name, self._sources)
		except RuntimeError:
			self.close()

		if not verbose:
			self.props.NO_USER_WARNING = True
			self.props.USER_LOG_P = False
		self.props.MAXIMUM_USER_ERROR = 42  # after this number of exceptions the programm will abort
		pypsutils.build_module_list(self)
		self._name=self.info("workspace")[0]

		"""Call all the functions 'post_init' of the given parents"""
		for pws in reversed(self.iparents):
			try:
				pws.post_init(sources, **kwargs)
			except AttributeError:
				pass

	def __enter__(self):
		"""handler for the with keyword"""
		return self
	def __exit__(self,exc_type, exc_val, exc_tb):
		"""handler for the with keyword"""
		if exc_type:# we want to keep info on why we aborted
			self.deleteOnClose=False 
		self.close()
		return False

	@property
	def name(self):return self._name


	def __iter__(self):
		"""provide an iterator on workspace's module, so that you can write
			map(do_something,my_workspace)"""
		return self._modules.itervalues()


	def __getitem__(self,module_name):
		"""retrieve a module of the module from its name"""
		pypsutils.build_module_list(self)
		return self._modules[module_name]


	def __setitem__(self,i):
		"""change a module of the module from its name"""
		return self._modules[i]


	def __contains__(self, module_name):
		"""Test if the workspace contains a given module"""
		pypsutils.build_module_list(self)
		return module_name in self._modules

	def info(self,topic):
		return split(self.cpypips.info(topic))

	def dirname(self):
		"""retrieve workspace database directory"""
		return self._name+".database/"

	def tmpdirname(self):
		"""retrieve workspace database directory"""
		return self.dirname()+"Tmp/"

	def checkpoint(self):
		"""checkpoints the workspace and returns a workspace id"""
		self.cpypips.close_workspace(0)
		chkdir=".{0}.chk{1}".format(self.dirname()[0:-1],len(self.checkpoints))
		shutil.copytree(self.dirname(), chkdir)
		self.checkpoints.append(chkdir)
		self.cpypips.open_workspace(self.name)
		return chkdir

	def restore(self,chkdir):
		self.props.PIPSDBM_RESOURCES_TO_DELETE = "all"
		self.cpypips.close_workspace(0)
		shutil.rmtree(self.dirname())
		shutil.copytree(chkdir, self.dirname())
		self.cpypips.open_workspace(self.name)


	def save(self, rep=None):
		"""save workspace back into source form in directory rep if given.
		By default, keep it into the .database/Src PIPS default"""
		self.cpypips.apply("UNSPLIT","%ALL")
		if rep:
			if not os.path.exists(rep):
				os.makedirs(rep)
			if not os.path.isdir(rep):
				raise ValueError("'{0}' is not a directory".format(rep))

		saved=[]
		for s in os.listdir(self.dirname()+"Src"):
			f = os.path.join(self.dirname(),"Src",s)
			if self.recoverInclude:
				# Recover includes on all the files.
				# Guess that nothing is done on Fortran files... :-/
				pypsutils.unincludes(f)
			if rep:
				# Save to the given directory if any:
				cp=os.path.join(rep,s)
				shutil.copy(f,cp)
				# Keep track of the file name:
				saved.append(cp)
			else:
				saved.append(f)

		return saved

	def compile(self,CC="gcc",CFLAGS="-O2 -g", LDFLAGS="", link=True, rep=None, outfile="",extrafiles=[]):
		"""try to compile current workspace with compiler `CC', compilation flags `CFLAGS', linker flags `LDFLAGS' in directory `rep' as binary `outfile' and adding sources from `extrafiles'"""
		if not rep:
			rep=self.tmpdirname()+"d.out"
		otmpfiles=self.save(rep=rep)+extrafiles
		command=[CC, self.cppflags, CFLAGS]
		if link:
			if not outfile:
				outfile=self._name
			self.goingToRunWith(otmpfiles, rep)
			command+=otmpfiles
			command+=[LDFLAGS]
			command+=["-o", outfile]
		else:
			self.goingToRunWith(otmpfiles, rep)
			command+=["-c"]
			command+=otmpfiles
		commandline = " ".join(command)
		if self.verbose:
			print >> sys.stderr , "Compiling the workspace with", commandline
		ret = os.system(commandline)
		if ret:
			if not link: map(os.remove,otmpfiles)
			raise RuntimeError("`%s' failed with return code %d" % (commandline, ret >> 8))
		return outfile


	def goingToRunWith(self, files, rep):
		""" hook that happens just before compilation of `files' in `rep'"""
		for f in files:
			pypsutils.addMAX0(f)
			# fix bad pragma pretty print
			lines=[]
			with open(f,"r") as source:
				pragma=""
				for line in source:
					if re.match(r'^#pragma .*',line):
						pragma=line
					elif re.match(r'^\w+:',line) and pragma:
						lines.append(line)
						lines.append(pragma)
						pragma=""
					else:
						lines.append(line)

			with open(f,"w") as source:
				source.writelines(lines)

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
		pypsutils.build_module_list(self)
		the_modules=[m for m in self._modules.values() if matching(m)]
		return modules(the_modules)


	# Create an "all" pseudo-variable that is in fact the execution of
	# filter with the default filtering rule: True
	all = property(filter)

	# Transform in the same way the filtered compilation units as a
	# pseudo-variable:
	compilation_units = property(pypsutils.filter_compilation_units)

	# Build also a pseudo-variable for the set of all the functions of the
	# workspace.  We should ask PIPS top-level for it instead of
	# recomputing it here, but it is so simple this way...
	all_functions = property(pypsutils.filter_all_functions)

	@staticmethod
	def delete(name):
		"""Delete a workspace by name

		It is a static method of the class since an open workspace
		cannot be deleted without creating havoc..."""
		pypips.delete_workspace(name)
		try: shutil.rmtree(pypsutils.nameToTmpDirName(name))
		except OSError: pass


	def close(self):
		"""force cleaning and deletion of the workspace"""
		map(shutil.rmtree,self.checkpoints)
		try : self.cpypips.close_workspace(0)
		except RuntimeError: pass
		if self.deleteOnClose:
			try : workspace.delete(self._name)
			except RuntimeError: pass
		if self.tmpDirName:
			try : shutil.rmtree(self.tmpDirName)
			except OSError: pass
		self.hasBeenClosed = True

	def __getattribute__(self, name):
		"""Method overriding the normal attribute access."""

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

		#Functions with '__' before can't be overloaded (for example
		# __dir__), but they can still have pre_ and post_ hooks.
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
		#		"""a method that compiles then launches gdb"""
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

		# If we didn't find anything, raise an error
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
		'''Allow user to access a compilation unit by writing w.cu.compilation_unit_name'''
		def __init__(self,wp):
			self.__dict__['_wp'] = wp

		def __setattr__(self, name, val):
			raise AttributeError("Compilation Unit assignement is not allowed.")

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
			"""provide an iterator on workspace's compilation unit, so that you can write
				map(do_something,my_workspace)"""
			return self._cuDict().itervalues()

		def __pyropsFakeIter__(self):
			"""provide an iterator on workspace's module, so that you can write
				map(do_something,my_workspace)"""
			return self._cuDict().values()

		def __getitem__(self,module_name):
			"""retrieve a module of the workspace from its name"""
			return self._cuDict()[module_name]

		def __setitem__(self,i):
			"""change a module of the workspace from its name"""
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
			"""provide an iterator on workspace's funtions, so that you
				can write map(do_something, my_workspace.fun)"""
			return self._functionDict().itervalues()

		def __pyropsFakeIter__(self):
			"""provide an iterator on workspace's funtions, so that you
				can write map(do_something, my_workspace.fun)"""
			return self._functionDict().values()

		def __getitem__(self,module_name):
			"""retrieve a module of the workspace by its name"""
			return self._functionDict()[module_name]
		def __setitem__(self,i):
			"""change a module of the workspace by its name"""
			return self._functionDict()[i]

		def __contains__(self, module_name):
			"""Test if the workspace contains a given module"""
			return module_name in self._functionDict()

	class props(object):
		"""Allow user to access a property by writing w.props.PROP,
		this class contains a static dictionnary of every properties
		and default value

		Provides also iterator and [] methods

		TODO : allows global setting of many properties at once

		all is a local dictionary with all the properties with their initial
		values. It is generated externally.
		"""
		def __init__(self,wp):
			self.__dict__['wp'] = wp

		def __setattr__(self, name, val):
			if name.upper() in self.all:
				pypsutils._set_property(self.wp,name,val)
			else:
				raise NameError("Unknow property : " + name)

		def __getattr__(self, name):
			if name.upper() in self.all:
				return pypsutils.get_property(self.wp,name)
			else:
				raise NameError("Unknow property : " + name)

		def __dir__(self):
			"We should use the updated values, not the default ones..."
			return self.all.keys()

		def __iter__(self):
			"""provide an iterator on workspace's properties, so that you
				can write map(do_something, my_workspace.props)"""
			return self.all.itervalues()

		def __pyropsFakeIter__(self):
			"""provide an iterator on workspace's module, so that you can write
				map(do_something,my_workspace)"""
			return self.all.values()

		def __getitem__(self, property_name):
			"""retrieve a property of the workspace by its name"""
			return self.__getattr__(self, property_name)

		def __setitem__(self, property_name, value):
			"""change the property of the workspace by its name"""
			self.__setattr__(property_name, value)

		def __contains__(self, property_name):
			"""Test if the workspace contains a given property"""
			return property_name in self.all

		# Here the source of the property all attribute will be generated:
### props_methods /!\ do not touch this line /!\

# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
