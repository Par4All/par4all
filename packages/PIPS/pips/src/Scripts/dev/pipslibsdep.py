#!/usr/bin/env python
#
# $Id$
#
# Copyright 1989-2010 MINES ParisTech
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
from os.path import isdir, isfile
from re import match, sub
from subprocess import Popen, PIPE
from sys import argv,stderr,exit

generate_dot_file=False
libdir="src/Libs"
srcdir=path.join("..","src/Libs")
if not isfile(path.join(srcdir,"syntax/syntax-local.h")):
	srcdir=argv[1]

print """\
=========================
PIPS include checker
by serge guelton o(^_-)O
=========================\
"""

class sym:
	def __init__(self,name):
		self.name=name
	def undefined(self):
		return match("^ +U (\w+)$",self.name)
	def defined(self):
		return match("^[0123456789abcdef]+ +[TD] (\w+)$",self.name)



class library:
	def __init__(self,name,path):
		self.name=name
		self.path=path
		self.used_symbols = set()
		self.defined_symbols = set()
		self.depends = []
	def objects(self):
		objsdir=path.join(self.path,".libs")
		if not isdir(objsdir):
			raise "object files not built in " + self.name
		objs=[ o for o in listdir(objsdir) if match(".*\.o$",o) ]
		if not objs:
			raise "object files not built in " + self.name
		return map(lambda x:path.join(objsdir,x),objs)
	def includes(self):
		depsdir=path.join(srcdir,self.name)
		if not isdir(depsdir):
			raise "dep files not built in " + depsdir
		deps=set()
		for d in listdir(depsdir):
			if match(".*\.[cly]$",d):
				fd=file(path.join(depsdir,d))
				for line in fd:
					m= match('#include\s*"(\w+)\.h"',line)
					if m:deps.add(m.groups()[0])
		return deps


	def dotname(self): return sub("-","_",self.name)
	def dotstr(self,full=True):
		fmt=self.dotname()
		fmt+= ' [label="'+self.name+'"]'
		fmt+='\n'
		if full:
			for d in self.depends:
				fmt+=d.dotname()
				fmt+=" -> "
				fmt+=self.dotname()
				fmt+='\n'
		return fmt



libraries = [ library(d,path.join(libdir,d)) for d in listdir(libdir) if isdir(path.join(libdir,d)) and isfile(path.join(libdir,d,"Makefile")) ]

for lib in libraries:

	for obj in lib.objects():
		cmd=["nm",obj]
		sp=Popen(cmd,stdout=PIPE)
		sp.wait()
		for symbol in sp.stdout.readlines():
			s=sym(symbol)
			m = s.undefined()
			if m:
				lib.used_symbols.add(m.groups()[0])
			else:
				m = s.defined()
				if m:
					lib.defined_symbols.add(m.groups()[0])




# l0 depends on l1 if l0.used_symbols inter l1.defined_symbols != 0
for lib0 in libraries:
	for lib1 in libraries:
		if lib0.used_symbols.intersection(lib1.defined_symbols):
			lib0.depends.append(lib1)

# some checks on includes
check_result=0
print "checking library includes"
for lib in libraries:
	sdepends=set(map(lambda x:x.name,lib.depends))
	diff=lib.includes().difference(sdepends)
	if len(argv) == 1 or lib.name in argv[1:]:
		for elib in diff:
			if elib in map(lambda x:x.name,libraries) and elib != lib.name:
				print >> stderr, elib,"included in",lib.name,"but never used" 
				check_result+=1

# some checks on definitions
for lib0 in libraries:
	for lib1 in libraries:
		if lib0 != lib1:
			for symbol in lib0.defined_symbols.intersection(lib1.defined_symbols):
				print >> stderr, lib0.name , "and", lib1.name, "both define symbol", symbol
				check_result+=1


			
			

# pretty print dot file if required
if generate_dot_file:
	# l0 depends on l1 if l0.used_symbols inter l1.defined_symbols != 0
	for lib0 in libraries:
		lib0.depends=list()
	for lib0 in libraries:
		for lib1 in libraries:
			s=lib0.defined_symbols.intersection(lib1.used_symbols)
			if s:
				#print >>stderr,lib0.name,s
				lib0.depends.append(lib1)

	print "digraph pipslibs {"
	if len(argv) == 1:
		for lib in libraries:
			print lib.dotstr()
	else:
		for lib in libraries:
			print lib.dotstr(lib.name in argv[1:])
	print "}"

if check_result == 0:
	print "everything ok"
exit(check_result)
