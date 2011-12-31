# -*- coding: utf-8 -*-
# Parse the benchmark configuration files and create according
# python object.

import os
import sac,pyps
import ConfigParser, copy
from ConfigParser import SafeConfigParser
ConfigParser.DEFAULTSECT='global'

LIST_SEP=":"

class Singleton(object):
	def __new__(cls, *args, **kwds):
		it = cls.__dict__.get("__it__")
		if it is not None:
			return it
		cls.__it__ = it = object.__new__(cls)
		it.init(*args, **kwds)
		return it
	def init(self, *args, **kwds): pass

class NeedOption(Exception): pass

class BenchSectionCfg(object):
	_attrs = {}
	_attrsVal = {}
	def __init__(self, sec_name, cfg_parser):
		self._cfg_parser = cfg_parser
		self._sec_name = sec_name

	def load(self):
		for name,v in self._cfg_parser.items(self._sec_name):
			if name in self._attrs:
				self._attrsVal[name] = self._attrs[name][0](v)
		# Set non-needed values to none if not specified
		for a,v in self._attrs.iteritems():
			if v[1] == False and a not in self._attrsVal:
				self._attrsVal[a] = None
		self._validate()
	
	def name(self):
		return copy.copy(self._sec_name)

	def get_list(*args, **kwargs): pass
	def __getattribute__(self, name):
		attrs = object.__getattribute__(self,"_attrs")
		attr = name.lower()
		if attr not in attrs:
			return object.__getattribute__(self,name)
		return self._attrsVal[attr]

	def __setattr__(self, name, v):
		attr = name.lower()
		if attr not in self._attrs:
			self.__dict__[name] = v
			return
		self._attrsVal[name] = v
		self._cfg_parser.set(self._sec_name, name, str(v))

	def __contains__(self, name):
		if name.lower() in self._attrsVal:
			if self._attrsVal[name.lower()] == None:
				return False
			return True
		return False

	def _validate(self):
		for attr,desc in self._attrs.iteritems():
			f,must = desc
			if must and attr not in self._attrsVal:
				raise NeedOption(attr)


class BenchSectionList(Singleton):
	_T = object
	def __init__(self):
		self._secs = dict()
	def set_parser(self, cfg_parser):
		self._cfg_parser = cfg_parser
		for sec in cfg_parser.sections():
			self._secs[sec] = self._T(sec, self._cfg_parser)
	def get(self, name):
		if name not in self._secs:
			self._secs[name] = self._T(name, self._cfg_parser)
		return self._secs[name]
	def list(self):
		return self._secs.keys()
	def save(self, filename):
		with open(filename, "w") as fp:
			self._cfg_parser.write(fp)

def _get_cls_section(clsList):
	def get_section(name):
		return clsList().get(name)
	return get_section

def make_list(T):
	def f(v):
		# TODO: filter '\LIST_SEP' in v
		return map(lambda x: T(x), v.split(LIST_SEP))
	return f

def make_maker(v):
	try:
		return getattr(pyps,v)
	except:
		return getattr(sac,v)


class cc(BenchSectionCfg):
	_attrs = {'cc': (unicode, True),
		  'cflags': (unicode, False),
		  'maker':(make_maker,True)}

class workspace(BenchSectionCfg):
	_attrs = {'files': (make_list(unicode), True),
		  'module': (unicode, True),
		  'args_validate': (unicode, True),
		  'args_benchmark': (unicode, True),
		  'iterations_bench': (int, True),
		  'include_dirs': (make_list(unicode), False),
		  'memalign': (bool, True),
		  'arg_kernel_size': (int, True)}

class remote_host(BenchSectionCfg):
	_attrs = {'host': (unicode, True), 'remote_working_dir': (unicode, True), 'control_path': (unicode, False)}


class ccList(BenchSectionList):
	_T = cc
class workspaceList(BenchSectionList):
	_T = workspace
class remoteHostList(BenchSectionList):
	_T = remote_host


class session(BenchSectionCfg):
	_attrs = {'cc_reference': (_get_cls_section(ccList), True),
		  'workspaces': (make_list(_get_cls_section(workspaceList)), True),
		  'default_mode': (unicode, True),
		  'default_driver': (unicode, True),
		  'default_remote': (_get_cls_section(remoteHostList), False),
		  'ccs_nosac': (make_list(_get_cls_section(ccList)), True),
		  'ccs_sac': (make_list(_get_cls_section(ccList)), False)}

class sessionList(BenchSectionList):
	_T = session

ccs = ccList()
workspaces = workspaceList()
sessions = sessionList()
remoteHosts = remoteHostList()

ccparser = SafeConfigParser()
sessionparser = SafeConfigParser()
workspaceparser = SafeConfigParser()
rhparser = SafeConfigParser()

def init(etc_dir, cc_file=None, session_file=None,wk_file=None,remote_file=None):
	''' Initialise the benchmark config module, using files from etc_dir. If one
	of the cc_file,session_file,wk_file or remote_file arguments are set, these
	files will be used insted of those in etc_dir/* '''
	def file_cfg(var, default):
		if var == None:
			return os.path.join(etc_dir, default)
		return var
	cc_file = file_cfg(cc_file, "ccs.cfg")
	session_file = file_cfg(session_file, "sessions.cfg")
	wk_file = file_cfg(wk_file, "workspaces.cfg")
	remote_file = file_cfg(remote_file, "remote_hosts.cfg")
	print session_file

	ccparser.read(cc_file)
	ccs.set_parser(ccparser)
	sessionparser.read(session_file)
	sessions.set_parser(sessionparser)
	workspaceparser.read(wk_file)
	workspaces.set_parser(workspaceparser)
	rhparser.read(remote_file)
	remoteHosts.set_parser(rhparser)

