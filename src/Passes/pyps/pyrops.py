import warnings
warnings.filterwarnings("ignore", message="the md5 module is deprecated; use hashlib instead")
import Pyro.core
import copy
import sys
import subprocess
import inspect

PyropsFile = inspect.getfile(inspect.currentframe())


class Launcher(Pyro.core.ObjBase):
	'''Class exported as a Pyro to get URI from child process'''
	daemon = None
	activeLaunchers = []

	def __init__(self):
		#Init Pyro if it hasn't been done yet
		if Launcher.daemon == None:
			Launcher.initServer()

		#Init object
		Pyro.core.ObjBase.__init__(self)
		self._uri = None 

		#Connect the new launcher
		uri = Launcher.daemon.connect(self,"launcher")
		
		#Launch child
		sp=subprocess.Popen([sys.executable,PyropsFile,str(uri)])

		Launcher.activeLaunchers.append(self)
		
		#Wait for child URI
		Launcher.daemon.requestLoop(lambda: self._uri == None, 0)
		
		#Retrieves Controller using the child URI
		self._ctrl =  Pyro.core.DynamicProxy(self._curi)

		#Retrieves Object using the child URI
		self._obj =  Pyro.core.DynamicProxy(self._uri)

	
	def setCURI(self,uri):
		self._curi = uri

	def setURI(self,uri):
		self._uri = uri
		self.exit=True

	def getObj(self):
		return self._obj

	@staticmethod
	def shutdown():
		a = copy.copy(Launcher.activeLaunchers)
		for i in a:
			i.close()
		Launcher.daemon.shutdown(True)

	def close(self):
		Launcher.activeLaunchers.remove(self)
		self._ctrl.shutdown()

	def __del__(self):
		self.close()

	@staticmethod
	def initServer():
		'''Start Pyro server'''
		if Launcher.daemon != None:
			return;

		Pyro.core.initServer(banner=0)
		Launcher.daemon=Pyro.core.Daemon()
	
class Control(Pyro.core.ObjBase):
	def __init__(self, daemon):
		Pyro.core.ObjBase.__init__(self)
		self.daemon = daemon
	
	def shutdown(self):
		self.daemon.shutdown()
		sys.stdout.flush()
		sys.stderr.flush()


import pyps

class pworkspace(pyps.workspace):
	def __init__(self,*args,**kwargs):
		self.launcher = Launcher()
		kwargs['cpypips']=self.launcher.getObj()
		super(pworkspace, self).__init__(*args, **kwargs)
       
	def close(self):
		pyps.workspace.close(self)
		self.launcher.close()

	def __del__(self):
		self.launcher.close()

def main():
	import pypips

	#Create a new server
	Pyro.core.initServer(banner=0)
	daemon=Pyro.core.Daemon()

	#Retrieve host launcher and give Workspace URI
	launcher = Pyro.core.getProxyForURI(sys.argv[1])

	#Create a new Object
	obj = Pyro.core.ObjBase()
	obj.delegateTo(pypips)
	uri = daemon.connect(obj,"ipypips")

	#Create a new Control
	curi = daemon.connect(Control(daemon),"control")

	launcher.setCURI(curi)
	launcher.setURI(uri)


	#Start Pyro request loop with a low timeout (0) for faster shutdown
	daemon.requestLoop(lambda: 1, 3)
	daemon.shutdown()

if __name__=="__main__":
	main()
