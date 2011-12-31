#!/usr/bin/env python
from __future__ import with_statement # to cope with python2.5
import pyps
import sys
import os
import shutil
import string
import tempfile
import pickle
import subprocess
from pyps import module

class object_code:
	"""preprocessed c source file descriptor"""
	def __init__(self,sourcefile,cppflags,cflags):
		self.cflags=cflags
		CPP=os.getenv("CPP","cpp")
		cmd=[CPP,"-U__GNUC__"]+cppflags+[sourcefile]
		#print "# running",cmd
		sp=subprocess.Popen(cmd,stdout=subprocess.PIPE)
		sp.wait()
		self.code=sp.stdout.read()
		self.cname=sourcefile.replace(os.sep,"__")
	def set_cname(self,cname):
		self.cname=cname
		for op in self.cflags:
			if op == "-c":
				i=self.cflags.index(op)
				self.cflags[i+1]=self.cname
				break
	def dump_to_c(self,in_dir):
		self.set_cname(in_dir+os.sep+self.cname)
		cfile=file(self.cname,"w")
		cfile.write(self.code)
		cfile.close()

def ofile(argv):
	for opt in argv[1:]:
		if opt == '-o':
			index=argv.index(opt)
			return argv[index+1]
	return ""

def cppflags(argv):
	flags=[]
	for opt in argv[1:]:
		if opt[0:2] == "-D" or opt[0:2] == "-I" :
			flags+=[opt]
			argv.remove(opt)
	return flags

class pipscc:
	"""modular pips compiler front-end"""
	def __init__(self,argv):
		"""create a pips compiler instance from argv"""
		self.argv=argv
		self.is_ld=len(self.gather_c_files())==0

	def run(self):
		"""run the compilation"""
		if not self.is_ld:self.pipscpp() 
		else:self.pipsld() 

	def pipscpp(self):
		"""simulate the behavior of the c preprocessor"""
		# parse command line
		CPPFLAGS=cppflags(self.argv)
		OUTFILE=ofile(self.argv)
		#print "# CPPFLAGS: ", CPPFLAGS
		cpp_and_linking= len([f for f in self.argv[1:] if f == "-c"]) == 0

		# look for input file
		for opt in self.argv[1:]:
			if opt[0] != '-' and opt[-2:] == '.c' :
				if not OUTFILE:
					OUTFILE=os.path.basename(opt)[0:-1]+"o"
				# generate internal representation of preprocessed code
				args = self.argv[1:]
				if cpp_and_linking : args.insert(0,"-c")
				obj=object_code(opt,CPPFLAGS,args)
				# serialize it
				newobj=file(OUTFILE,"w")
				pickle.dump(obj,newobj)
				newobj.close()
				#print "# OBJ written: ", OUTFILE
		# check if we should link too
		if cpp_and_linking:
			for i in range(1,len(self.argv)):
				if self.argv[i][-2:]=='.c':
					self.argv[i]=self.argv[i][0:-2]+".o"
			self.pipsld()
		# that's all folks

	def gather_object_files(self):
		INPUT_FILES=[]
		for opt in self.argv[1:]:
			if opt[0] != '-' and opt[-2:]==".o":
				INPUT_FILES+=[opt]
		return INPUT_FILES

	def gather_c_files(self):
		INPUT_FILES=[]
		for opt in self.argv[1:]:
			if opt[0] != '-' and opt[-2:]==".c":
				INPUT_FILES+=[opt]
		return INPUT_FILES

	def unpickle(self,WDIR,files):
		"""generate a list of unpickled object files from files"""
		O_FILES=[]
		for ifile in files:
				obj=pickle.load(file(ifile,"r"))
				obj.dump_to_c(WDIR)
				obj.oname=ifile
				O_FILES+=[obj]
		return O_FILES

	def changes(self,ws):
		"""apply any change to the workspace, should be overloaded by the user"""
		for f in ws.fun: f.display()
		for c in ws.cu: c.display()

	def get_wd(self):
		"""selects a working directory for pipscc"""
		WDIR=tempfile.mkdtemp("pipscc")
		#print "# intermediate files generated in", WDIR
		return WDIR

	def get_workspace(self,c_files):
		return pyps.workspace(*c_files)

	def compile(self,wdir,o_files):
		CC=os.getenv("CC","gcc")
		for obj in o_files:
			cmd=[CC]+obj.cflags+["-o",obj.oname]
			#print "# running", cmd
			sp=subprocess.Popen(cmd)
			sp.wait()
		
		cmd=[CC]+self.argv[1:]
		#print "# running", cmd
		sp=subprocess.Popen(cmd)
		exitcode=sp.wait()
		if exitcode:
			shutil.rmtree(wdir)

	def pipsld(self):
		"""simulate c linker, all computation is done at link time"""
		WDIR=self.get_wd()
		
		# gather pickled input files
		INPUT_FILES=self.gather_object_files()
		if len(INPUT_FILES) == 0:
			print >> sys.stderr, "pipscc: no input files"
			sys.exit(1)
		else:
			# load pickled input files
			O_FILES=self.unpickle(WDIR,INPUT_FILES)
			C_FILES=map(lambda o:o.cname,O_FILES)
			#print "# input files: ", C_FILES
			
			# run pips with this informations
			#print "# running pips"
			with self.get_workspace(C_FILES) as ws:
				# add extra operations 
				self.changes(ws)
				# commit changes
				ws.save(rep=WDIR)
			
			# now run the compiler
			self.compile(WDIR,O_FILES)
		shutil.rmtree(WDIR)

#
##
#

if __name__ == "__main__":
	thecompiler=pipscc(sys.argv)
	thecompiler.run()
