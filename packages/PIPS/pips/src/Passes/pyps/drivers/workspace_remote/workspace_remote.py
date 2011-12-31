from __future__ import with_statement # to cope with python2.5
import pyps
import pypsutils
import os
import sys
from subprocess import Popen, PIPE

class remoteExec:
    def __init__(self, host, remoteDir, controlMasterPath=None):
        self.__host = host
        self.__cmPath = controlMasterPath
        self.__remoteDir = remoteDir

        self.__ssh_opt = ""
        self.__use_cm = False
    
    def __enter__(self):
        return self

    def __exit__(self,exc_type, exc_val, exc_tb):
        self.del_cm()

    def del_cm(self):
        if not self.__use_cm: return
        os.system("ssh \"-oControlPath=%s\" -O exit none" % (self.__cmPath) )
        self.__use_cm = False

    def __init_cm(self):
        """ Initialise the SSH control master if necessary. """
        if self.__cmPath != None and not self.__use_cm:
            ret = os.system("ssh -f -N -oControlMaster=yes \"-oControlPath=%s\" %s" % (self.__cmPath, self._host))
            if ret != 0:
                raise RuntimeError("SSH: error while creating control master : ssh returned %d." % ret)
            self.__ssh_opt = "-oControlMaster=auto \"-oControlPath=%s\"" % self.__cmPath
            self.__use_cm = True
    
    def __get_ssh_cmd(self):
        return "ssh %s \"%s\"" % (self.__ssh_opt, self.__host)

    def do(self, cmd):
        """ Execute a remote command, and return a pipe to it. """
        self.__init_cm()
        cmd = "%s %s" % (self.__get_ssh_cmd(), cmd)
        return os.system(cmd)

    def doPopen(self, args, shell=False):
        """ Execute a remote command, and return a pipe to it. """
        self.__init_cm()
        nargs = ["ssh"]
        opts = self.__ssh_opt.split(' ')
        if not (len(opts) == 1 and opts[0] == ""):
            nargs.extend(opts)
        nargs.append(self.__host)
        #need one more level of quotes
        nargs.extend(map(lambda x: '"'+x+'"',args))
        print >> sys.stderr, nargs
        return Popen(nargs, shell=shell, stdout = PIPE, stderr = PIPE) 


    def copy(self, path, to):
        self.__init_cm()
        ret = os.system("scp %s %s %s:%s" % (self.__ssh_opt, path, self.__host, os.path.join(self.__remoteDir, to)))
        if ret != 0:
            raise RuntimeError("SSH: error while copying %s to %s:%s : scp returned %d." % (path, self.__host, to, ret))
    
    def copyRemote(self, pfrom, lto):
        self.__init_cm()
        ret = os.system("scp %s %s:%s %s" % (self.__ssh_opt, self.__host, os.path.join(self.__remoteDir,pfrom), lto))
        if ret != 0:
            raise RuntimeError("SSH: error while copying %s:%s to %s : scp returned %d." % (self.__host, pfrom, lto, ret ))

    def copyRec(self, path, to):
        self.__init_cm()
        print ("scp -r %s %s %s:%s" % (self.__ssh_opt, path, self.__host, os.path.join(self.__remoteDir, to)))
        ret = os.system("scp -r %s %s %s:%s" % (self.__ssh_opt, path, self.__host, os.path.join(self.__remoteDir, to)))
        if ret != 0:
            raise RuntimeError("SSH: error while copying %s to %s:%s : scp returned %d." % (path, self.__host, to, ret))

    def working_dir(self): return self.__remoteDir
    def host(self): return self.__host


class workspace(pyps.workspace):
    def __init__(self, *sources, **kwargs):
        self.__remoteExec = kwargs["remoteExec"]
        super(workspace,self).__init__(*sources,**kwargs)

    def save(self,rep=None):
        if rep == None:
            rep = self.tmpdirname
        files,headers = super(workspace,self).save(rep)    
        #setting otmpfiles for remote
        
        #setting rep for remote and cleaning it
        rdir = os.path.join(self.__remoteExec.working_dir(), rep)
        self.__remoteExec.do("rm -rf \"%s\"" % rdir)
        self.__remoteExec.do("mkdir -p \"%s\"" % rdir)
        
        # auxiliary function to copy files to host
        def rcopy(files):
            otmpfiles=list()
            for f in files:
                dst = os.path.join(rdir, os.path.basename(f))
                otmpfiles.append(dst)
                if self.verbose:
                    print >>sys.stderr, "Copy %s to remote %s..." % (f, rep)
                self.__remoteExec.copy(f, rep)
            return otmpfiles

        if self.verbose:
            print >> sys.stderr, "Copying files to remote host %s..." % self.__remoteExec.host()
        return rcopy(files),rcopy(headers)

    def divert(self,rep=None, maker=pyps.Maker()):
        if rep ==None:
            rep = self.tmpdirname
        makefile,others = super(workspace,self).divert(rep,maker)
        self.__remoteExec.copy(os.path.join(rep,makefile),rep)
        for f in others:
            self.__remoteExec.copy(os.path.join(rep,f),rep)
        return makefile,others
        
    def compile(self, maker=pyps.Maker(), rep=None, outfile="a.out", rule="all" ,**opts):
        """ Uses makefiles on the remote host to compile the workspace"""
        if rep == None:
            rep = self.tmpdirname
        self.divert(rep,maker)

        rep = os.path.join(self.__remoteExec.working_dir(),rep)
        commandline = pypsutils.gen_compile_command(rep,maker.makefile,outfile,rule,**opts)

        if self.verbose:
            print >> sys.stderr , "Compiling the remote workspace with", commandline
        #We need to set shell to False or it messes up with the make command
        p = self.__remoteExec.doPopen(commandline)
        (out,err) = p.communicate()
        rc = p.returncode
        if rc != 0:
            print >> sys.stderr, err
            raise RuntimeError("%s failed with return code %d" % (commandline, rc))

        return os.path.join(rep,outfile)

    def run(self, binary, args=[]):
        cmd = [os.path.join("./",binary)] + args
        p = self.__remoteExec.doPopen(cmd)
        (out,err) = p.communicate()
        rc = p.returncode
        if rc != 0:
            print >> sys.stderr, err
            raise RuntimeError("%s failed with return code %d" % (cmd, rc))
        return (rc,out,err)

