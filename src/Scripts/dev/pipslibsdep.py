#!/usr/bin/env python
#
# $Id$
#
# Copyright 1989-2014 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#

# Serge Guelton
# this quick'n dirty script generates on standard output a dot view of dependencies betwwen libraries
# it also prints on stderr a list of libraries that are uselessly #included by others
# parameters can be given: they are library names that must appear on the dependency check / view.
# If none is given, all libraries are considered
# you must manually set generate_dot_file to true if you want to activate this feature


from os import listdir, path
from os.path import isdir, isfile, basename
from re import match, sub
from subprocess import Popen, PIPE
from sys import argv,stderr,exit
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-s","--srcdir",
		action="store",type="string",dest="srcdir",
		metavar="<dir>", help="use <dir> as top_srcdir dir")

parser.add_option("-b","--builddir",
		action="store",type="string",dest="builddir",
		metavar="<dir>", help="use <dir> as top_builddir dir")

parser.add_option("-l","--lib",
		action="store",type="string",dest="libname",
		metavar="<libname>", help="only inspect <libname> instead of all libs")

parser.add_option("-d","--dot",
		action="store",type="string",dest="dotfile",
		metavar="<file>", help="generate dot description of dependencies in <file>")
parser.add_option("-v","--verbose",
		action="count", dest="verbose",default=0,
		help="make lots of noise (cumulative)")

(options, args) = parser.parse_args()

if not options.srcdir or not options.builddir:
	parser.error("options --srcdir and --builddir are mandatory")

libdir=path.join(options.builddir,"src","Libs")
srcdir=path.join(options.srcdir,"src","Libs")
if not isfile(path.join(srcdir,"syntax","syntax-local.h")):
	parser.error("options --srcdir does not point to pips source dir")
if not isfile(path.join(libdir,"syntax","syntax.h")):
	parser.error("options --builddir does not point to pips build dir")


print """\
=========================
PIPS include checker
by serge guelton o(^_-)O
=========================\
"""

def log(lvl,*msg):
	if lvl<= options.verbose:
		print reduce(lambda x,y : str(x)+" " +str(y) ,msg,"")

def dotname(name): return sub("-","_",name)

class sym:
	def __init__(self,name):
		self.name=name
	def undefined(self):
		m= match("^ +U (\w+)$",self.name)
		if m: return m.groups()[0]
		else: return None
	def defined(self):
		m=match("^[0123456789abcdef]+ +[TDB] (\w+)$",self.name)
		if m: return m.groups()[0]
		else: return None



class library:
	def __init__(self,name,path):
		self.name=name
		self.path=path
		self.objs=[]
		self.used_symbols = {}
		self.defined_symbols = {}
		self.used_libraries = {}

	def objects(self):
		if not self.objs:
			objsdir=path.join(self.path,".libs")
			if not isdir(objsdir):
				raise "object files not built in " + self.name
			objs=[ o for o in listdir(objsdir) if match(".*\.o$",o) ]
			if not objs:
				raise "object files not built in " + self.name
			self.objs= map(lambda x:path.join(objsdir,x),objs)
		return self.objs

	def compute_symbols(self):
		log(1, "computing symbols for", self.name)
		for obj in self.objects():
			log(2, "creating",self.name,"lib with entry",basename(obj))
			self.used_libraries[basename(obj)]=set()
			cmd=["nm",obj]
			sp=Popen(cmd,stdout=PIPE)
			sp.wait()
			for symbol in sp.stdout.readlines():
				s=sym(symbol)
				m = s.undefined()
				if m:
					log(2, m , "used by", basename(obj), "in",self.name)
					if not lib.used_symbols.has_key(m):
						lib.used_symbols[m]=set()
					lib.used_symbols[m].add(basename(obj))
				else:
					m = s.defined()
					if m:
						log(2, m, "defined by", basename(obj), "in",self.name)
						lib.defined_symbols[m]=basename(obj)

	def compute_deps(self,alllibs):
		log(1, "computing dependencies for", self.name)
		for (sym,objs) in self.used_symbols.iteritems():
			log(2, "checking symbol", sym, "in objects" , objs, "and lib", self.name)
			for otherlib in alllibs:
				if otherlib != self:
					if sym in otherlib.defined_symbols.keys():
						log(2,sym, "from",otherlib.name, "used in" , self.name, "by", objs)
						for obj in objs:
							self.used_libraries[obj].add(otherlib.name)
		return self.used_libraries

	def check_includes(self,alllibs):
		depsdir=path.join(srcdir,self.name)
		err={}
		if not isdir(depsdir):
			raise "dep files not built in " + depsdir
		for d in listdir(depsdir):
			log(2,"checking includes for", self.name, ":", d)
			err[d]=set()
			if match(".*\.[cly]$",d):
				fd=file(path.join(depsdir,d))
				for line in fd:
					m= match('#include\s*"(\w+)\.h"',line)
					if m: # d depends on (m.groups()[0])
						deplib = m.groups()[0]
						if deplib in map(lambda x:x.name,alllibs) and deplib != self.name:
							obj=d[:-1]+"o"
							if self.used_libraries.has_key(obj) and deplib not in self.used_libraries[obj]:
								err[d].add(deplib)
		return err


	def dotstr(self,only=None):
		fmt=dotname(self.name)
		fmt+= ' [label="'+self.name+'"]'
		fmt+='\n'
		if not only or only == self.name:
			for d in reduce(lambda x,y:x.union(y) ,self.used_libraries.values(),set()):
				fmt+=dotname(d)
				fmt+=" -> "
				fmt+=dotname(self.name)
				fmt+='\n'
		return fmt



libraries = [ library(d,path.join(libdir,d)) for d in listdir(libdir) if isdir(path.join(libdir,d)) and isfile(path.join(libdir,d,"Makefile")) ]

for lib in libraries:
	lib.compute_symbols()

for lib in libraries:
	lib.compute_deps(libraries)

check_result = 0
for lib in libraries:
	if not options.libname or lib.name == options.libname:
		err = lib.check_includes(libraries)
		for (k,v) in err.iteritems():
			if v:
				check_result+=1
				print >> stderr, lib.name,":",k ,":", reduce(lambda x,y:x+" "+y,v,"")


# pretty print dot file if required
if options.dotfile:
	fd=file(options.dotfile,"w")
	print >> fd,"digraph pipslibs {"
	if options.libname:
		for lib in libraries:
			print >> fd , lib.dotstr(options.libname)
	else:
		for lib in libraries:
			print >> fd, lib.dotstr()
	print >> fd , "}"
	fd.close()

if check_result == 0:
	print "everything ok"
else:
	print check_result, "errors"
exit(check_result)
