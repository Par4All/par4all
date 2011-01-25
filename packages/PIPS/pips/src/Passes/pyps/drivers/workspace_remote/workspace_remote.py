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


class workspace:
	def __init__(self, ws, source, *args, **kwargs):
		self.ws = ws
		self.remoteExec = kwargs["remoteExec"]

#	def compile(self,CC="gcc",CFLAGS="-O2 -g", LDFLAGS="", link=True, rep="d.out", outfile="",extrafiles=[],*args,**kwargs):
	def compile(self,ccexecp,*args,**kwargs):
		"""try to compile current workspace with compiler `CC', compilation flags `CFLAGS', linker flags `LDFLAGS' in directory `rep' as binary `outfile' and adding sources from `extrafiles'"""

		CC = ccexecp.CC
		CFLAGS = ccexecp.CFLAGS
		LDFLAGS = ccexecp.LDFLAGS
		link = kwargs.get("link", True)
		outfile = ccexecp.outfile
		extrafiles = ccexecp.extrafiles

		if ccexecp.rep==None:
			ccexecp.rep=self.ws.tmpdirname()+"d.out"

		otmpfiles=self.ws.save(rep=ccexecp.rep)+extrafiles
		otmpfilesrem = []
		

		rdir = os.path.join(self.remoteExec.working_dir(), ccexecp.rep)
		self.remoteExec.do("rm -rf \"%s\"" % rdir)
		self.remoteExec.do("mkdir -p \"%s\"" % rdir)
		

		command=[CC, self.ws.cppflags, CFLAGS]
		if link:
			if not outfile:
				outfile=self.ws._name
			outfile = os.path.join(self.remoteExec.working_dir(),outfile)
			self.ws.goingToRunWith(otmpfiles, ccexecp.rep)
			for f in otmpfiles:
				dst = os.path.join(rdir, f[len(ccexecp.rep)+1:])
				otmpfilesrem.append(dst)
			command+=otmpfilesrem
			command+=[LDFLAGS]
			command+=["-o",outfile]
		else:
			self.ws.goingToRunWith(otmpfiles, ccexecp.rep)
			for f in otmpfiles:
				dst = os.path.join(rdir, f[len(ccexecp.rep)+1:])
				otmpfilesrem.append(dst)
			command+=["-c"]
			command+=otmpfilesrem
		commandline = " ".join(command)

		# Copy files to remote host
		if self.ws.verbose:
			print >> sys.stderr, "Copying files to remote host %s..." % self.remoteExec.host()
		for f in extrafiles:
			self.remoteExec.copy(f, rdir)

		rtmp = os.path.split(rdir)[0]
		print >>sys.stderr, "Copy recursive %s to remote %s..." % (ccexecp.rep, rtmp)
		self.remoteExec.copyRec(ccexecp.rep, rtmp)

		if self.ws.verbose:
			print >> sys.stderr , "Compiling the workspace with", commandline

		p = self.remoteExec.doPopen(command)
		(out,err) = p.communicate()
		ccexecp.cc_stderr = err
		ret = p.returncode
		if ret != 0:
			if not link: map(os.remove,otmpfiles)
			print >> sys.stderr, err
			raise RuntimeError("%s failed with return code %d" % (commandline, ret))

		ccexecp.outfile = outfile
		ccexecp._compile_done = True
		ccexecp.cc_cmd = commandline
		ccexecp.cmd = [ccexecp.outfile] + ccexecp.args
		return outfile

	def run_output(self, ccexecp):
		if not ccexecp.cmd:
			return self.compile_and_run(ccexecp)
		p = self.remoteExec.doPopen(ccexecp.cmd)
		(out,err) = p.communicate()
		rc = p.returncode
		return (rc,out,err)

