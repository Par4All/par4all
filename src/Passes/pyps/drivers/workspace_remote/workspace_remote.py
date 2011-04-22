from __future__ import with_statement # to cope with python2.5
import pyps
import os
import sys
from subprocess import Popen, PIPE

class remoteExec:
	def __init__(self, host, controlMasterPath=None, remoteDir=None):
		self._host = host
		self._cmPath = controlMasterPath
		self._remoteDir = remoteDir

		self._ssh_opt = ""
		self._use_cm = False
	
	def __enter__(self):
		return self

	def __exit__(self,exc_type, exc_val, exc_tb):
		self.del_cm()

	def del_cm(self):
		if not self._use_cm: return
		os.system("ssh \"-oControlPath=%s\" -O exit none")
		self._use_cm = False

	def init_cm(self):
		""" Initialise the SSH control master if necessary. """
		if self._cmPath != None and not self._use_cm:
			ret = os.system("ssh -f -N -oControlMaster=yes \"-oControlPath=%s\" %s" % (self._cmPath, self._host))
			if ret != 0:
				raise RuntimeError("SSH: error while creating control master : ssh returned %d." % ret)
			self._ssh_opt = "-oControlMaster=auto \"-oControlPath=%s\"" % self._cmPath
			self._use_cm = True
	
	def get_ssh_cmd(self):
		return "ssh %s \"%s\"" % (self._ssh_opt, self._host)

	def do(self, cmd):
		""" Execute a remote command, and return a pipe to it. """
		self.init_cm()
		cmd = "%s %s" % (self.get_ssh_cmd(), cmd)
		return os.system(cmd)

	def doPopen(self, args, shell=False):
		""" Execute a remote command, and return a pipe to it. """
		self.init_cm()
		nargs = ["ssh"]
		opts = self._ssh_opt.split(' ')
		if not (len(opts) == 1 and opts[0] == ""):
			nargs.extend(opts)
		nargs.append(self._host)
		nargs.extend(args)
		return Popen(nargs, shell=shell, stdout = PIPE, stderr = PIPE) 


	def copy(self, path, to):
		self.init_cm()
		ret = os.system("scp %s %s %s:%s" % (self._ssh_opt, path, self._host, os.path.join(self._remoteDir, to)))
		if ret != 0:
			raise RuntimeError("SSH: error while copying %s to %s:%s : scp returned %d." % (path, self._host, to, ret))
	
	def copyRemote(self, pfrom, lto):
		self.init_cm()
		ret = os.system("scp %s %s:%s %s" % (self._ssh_opt, self._host, os.path.join(self._remoteDir,pfrom), lto))
		if ret != 0:
			raise RuntimeError("SSH: error while copying %s:%s to %s : scp returned %d." % (self._host, pfrom, lto, ret ))

	def copyRec(self, path, to):
		self.init_cm()
		ret = os.system("scp -r %s %s %s:%s" % (self._ssh_opt, path, self._host, os.path.join(self._remoteDir, to)))
		if ret != 0:
			raise RuntimeError("SSH: error while copying %s to %s:%s : scp returned %d." % (path, self._host, to, ret))

	def working_dir(self): return self._remoteDir
	def host(self): return self._host


class remoteCompiler(pyps.backendCompiler):
	def __init__(self,CC="cc",CFLAGS="",LDFLAGS="",compilemethod=None,rep=None,outfile="", args=[], remoteExec=None):
		self.remoteExec = remoteExec
		
		#setting rep for remote and cleaning it
		self.rdir = os.path.join(self.remoteExec.working_dir(), compiler.rep)
		self.remoteExec.do("rm -rf \"%s\"" % rdir)
		self.remoteExec.do("mkdir -p \"%s\"" % rdir)
				
		super(remoteCompiler,self).__init__(CC, CFLAGS,LDFLAGS,compilemethod,rep,outfile,args)

	def compile(self, filename, extraCFLAGS="", verbose=False)
		"""compiles one given .c file. Called by the workspace method compile"""
		outfilename = os.path.basename(os.path.splitext(filename)[0]+".o")
		outfilename = os.path.join(self.remoteExec.working_dir(),outfile)
		command = [self.CC, extraCFLAGS, self.CFLAGS, "-c", filename, "-o", outfilename]
		commandline = " ".join(command)
		if verbose:
			print >> sys.stderr, "Copying file %s to remote host %s..." % filename % self.remoteExec.host()
		self.remoteExec.copy(f, rdir)

		if verbose:
			print >> sys.stderr , "Compiling a workspace file with", commandline
		
		p = self.remoteExec.doPopen(command)
		(out,err) = p.communicate()
		compiler.cc_stderr = err
		ret = p.returncode
		if ret != 0:
			print >> sys.stderr, err
			raise RuntimeError("%s failed with return code %d" % (commandline, ret))

		self.cc_cmd = commandline
		return [outfilename]

	def link(self,files, verbose=False)
		"""links a given list of .o files. Called by the workspace method compile"""
		command = [self.CC]+files+[self.LDFLAGS]
		if self.outfile:
			command+=["-o", self.outfile]
		commandline = " ".join(command)
		if verbose:
			print >> sys.stderr , "Linking the workspace with", commandline
		p = Popen(commandline, shell=True, stdout = PIPE, stderr = PIPE)
		(out,err) = p.communicate()
		self.cc_stderr = err
		ret = p.returncode
		if ret != 0:
			print >> sys.stderr, err
			raise RuntimeError("%s failed with return code %d" % (commandline, ret))
		self.cc_cmd = commandline
		return self.outfile







class workspace:
	def __init__(self, ws, *args, **kwargs):
		self.ws = ws
		self.remoteExec = kwargs["remoteExec"]

#	def compile(self,CC="gcc",CFLAGS="-O2 -g", LDFLAGS="", link=True, rep="d.out", outfile="",extrafiles=[],*args,**kwargs):
	def compile(self,compiler,*args,**kwargs):
		"""try to compile current workspace with compiler `CC', compilation flags `CFLAGS', linker flags `LDFLAGS' in directory `rep' as binary `outfile' and adding sources from `extrafiles'"""

		#setting flags in workspace
		CC = compiler.CC
		CFLAGS = compiler.CFLAGS
		LDFLAGS = compiler.LDFLAGS
		link = kwargs.get("link", True)
		outfile = compiler.outfile
		#setting rep in workspace
		if compiler.rep==None:
			compiler.rep=self.ws.tmpdirname()+"d.out"
		
		#setting otmpfiles in workspace
		otmpfiles=self.ws.save(rep=compiler.rep)
		#setting otmpfiles for remote
		otmpfilesrem = []
		
		#setting rep for remote and cleaning it
		rdir = os.path.join(self.remoteExec.working_dir(), compiler.rep)
		self.remoteExec.do("rm -rf \"%s\"" % rdir)
		self.remoteExec.do("mkdir -p \"%s\"" % rdir)
	
		#copy all headers in remote
		print >>sys.stderr, "Copying user headers to remote host %s..." % self.remoteExec.host()
		user_headers = self.ws.user_headers()
		for uh in user_headers:
			# TODO: try to find out in which directory the headers should go...
			#rrep_inc = os.path.join(rdir, os.path.dirname(uh))
			#self.remoteExec.do("mkdir -p \"%s\"" % rrep_inc)
			self.remoteExec.copy(uh, rdir)
	
		#setting command 
		command=[CC, self.ws.cppflags, CFLAGS]
		if link:
			if not outfile:
				outfile=self.ws._name
			outfile = os.path.join(self.remoteExec.working_dir(),outfile)
			self.ws.goingToRunWith(otmpfiles, compiler.rep)
			for f in otmpfiles:
				dst = os.path.join(rdir, f[len(compiler.rep)+1:])
				otmpfilesrem.append(dst)
			command+=otmpfilesrem
			command+=[LDFLAGS]
			command+=["-o",outfile]
		else:
			for f in otmpfiles:
				dst = os.path.join(rdir, f[len(compiler.rep)+1:])
				otmpfilesrem.append(dst)
			command+=["-c"]
			command+=otmpfilesrem
		commandline = " ".join(command)

		# Copy files to remote host
		if self.ws.verbose:
			print >> sys.stderr, "Copying files to remote host %s..." % self.remoteExec.host()
		rtmp = os.path.split(rdir)[0]
		print >>sys.stderr, "Copy recursive %s to remote %s..." % (compiler.rep, rtmp)
		self.remoteExec.copyRec(compiler.rep, rtmp)

		if self.ws.verbose:
			print >> sys.stderr , "Compiling the workspace with", commandline

		p = self.remoteExec.doPopen(command)
		(out,err) = p.communicate()
		compiler.cc_stderr = err
		ret = p.returncode
		if ret != 0:
			if not link: map(os.remove,otmpfiles)
			print >> sys.stderr, err
			raise RuntimeError("%s failed with return code %d" % (commandline, ret))

		compiler.outfile = outfile
		compiler._compile_done = True
		compiler.cc_cmd = commandline
		return outfile

	def run_output(self, compiler):
		if not compiler._compile_done:
			return self.ws.compile_and_run(compiler)
		compiler.cmd = [compiler.outfile] + compiler.args
		p = self.remoteExec.doPopen(compiler.cmd)
		(out,err) = p.communicate()
		rc = p.returncode
		return (rc,out,err)

