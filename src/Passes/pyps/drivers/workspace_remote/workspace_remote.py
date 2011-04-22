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
		print ("scp -r %s %s %s:%s" % (self._ssh_opt, path, self._host, os.path.join(self._remoteDir, to)))
		ret = os.system("scp -r %s %s %s:%s" % (self._ssh_opt, path, self._host, os.path.join(self._remoteDir, to)))
		if ret != 0:
			raise RuntimeError("SSH: error while copying %s to %s:%s : scp returned %d." % (path, self._host, to, ret))

	def working_dir(self): return self._remoteDir
	def host(self): return self._host


class workspace(pyps.workspace):
	def __init__(self, *sources, **kwargs):
		self.remoteExec = kwargs["remoteExec"]
		super(workspace,self).__init__(*sources,**kwargs)

	def save(self,rep=None):
		if rep == None:
			rep = self.tmpdirname()
		files = super(workspace,self).save(rep)	
		#setting otmpfiles for remote
		otmpfilesrem = []
		
		#setting rep for remote and cleaning it
		rdir = os.path.join(self.remoteExec.working_dir(), rep)
		self.remoteExec.do("rm -rf \"%s\"" % rdir)
		self.remoteExec.do("mkdir -p \"%s\"" % rdir)

		#copy all headers in remote
		#print >>sys.stderr, "Copying user headers to remote host %s..." % self.remoteExec.host()
		#user_headers = self.user_headers()
		#for uh in user_headers:
			# TODO: try to find out in which directory the headers should go...
			#rrep_inc = os.path.join(rdir, os.path.dirname(uh))
			#self.remoteExec.do("mkdir -p \"%s\"" % rrep_inc)
		#	self.remoteExec.copy(uh, rdir)
		
		if self.verbose:
			print >> sys.stderr, "Copying files to remote host %s..." % self.remoteExec.host()
		#rtmp = os.path.split(rep)[0]
		
		print files
		for f in files:
			dst = os.path.join(rdir, os.path.basename(f))
			otmpfilesrem.append(dst)
			print >>sys.stderr, "Copy %s to remote %s..." % (f, rep)
			self.remoteExec.copy(f, rep)

		return otmpfilesrem

	def make(self,rep=None, maker=pyps.Maker()):
		if rep ==None:
			rep = self.tmpdirname()
		makefile = super(workspace,self).make(rep,maker)
		#rtmp = os.path.split(rep)[0]
		self.remoteExec.copy(os.path.join(rep,makefile),rep)
		return makefile
		
	def compile(self,rep=None, makefile="Makefile", outfile="a.out"):
		""" Uses makefiles on the remote host to compile the workspace"""
		if rep == None:
			rep = self.tmpdirname()
		rep = os.path.join(self.remoteExec.working_dir(),rep)
		commandline = ["make",]
		commandline+=["-C",rep]
		commandline+=["-f",makefile]
		commandline.append("TARGET="+outfile)
		
		if self.verbose:
			print >> sys.stderr , "Compiling the remote workspace with", commandline
		#We need to set shell to False or it messes up with the make command
		p = self.remoteExec.doPopen(commandline)
		(out,err) = p.communicate()
		rc = p.returncode
		if rc != 0:
			print >> sys.stderr, err
			raise RuntimeError("%s failed with return code %d" % (commandline, rc))

		return os.path.join(rep,outfile),rc,out,err

	def run(self, binary, args=[]):
		cmd = [os.path.join("./",binary)] + args
		p = self.remoteExec.doPopen(cmd)
		(out,err) = p.communicate()
		rc = p.returncode
		if rc != 0:
			print >> sys.stderr, err
			raise RuntimeError("%s failed with return code %d" % (cmd, rc))
		return (rc,out,err)

