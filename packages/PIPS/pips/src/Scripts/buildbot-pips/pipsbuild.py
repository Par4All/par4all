from buildbot.buildslave import BuildSlave
from buildbot.process import factory
from buildbot.steps.source import SVN, Git
from buildbot.steps.shell import Configure,Compile,ShellCommand
from buildbot.steps.transfer import FileUpload,FileDownload
from autoconf import GNUAutoconf, Cleaning
from pipsvalid import PipsValid, GetPipsTarball, GetPipsSvn
from buildbot.scheduler import Scheduler, Periodic, Nightly
import os

class Pipsbuild(object):

	def __init__(self):
		self.pips_archs = {
			"linux-64":"ilun-x46",
			"bsd-64":"sb-d46",
#			"mac-32":"am-c23",
		}

		self.www = "/var/www/pips/"
		self.install_prefix='/tmp/pips_root/'
		self.configure_flags = ['--disable-static','--prefix='+self.install_prefix,"PKG_CONFIG_PATH="+self.install_prefix+"lib/pkgconfig","PATH=/sbin:"+self.install_prefix+"bin:/usr/local/bin:"+os.environ['PATH'],"CPPFLAGS=-I/usr/local/include",'--enable-doc', "PYTHON_VERSION=2.6"]
		self.dist_install_prefix='/tmp/dist/pips_root/'
		self.dist_configure_flags=['--disable-static','--prefix='+self.dist_install_prefix,"PKG_CONFIG_PATH="+self.install_prefix+"lib/pkgconfig","PATH=/sbin:"+self.install_prefix+"bin:/usr/local/bin:"+os.environ['PATH'],"CPPFLAGS=-I/usr/local/include", "PYTHON_VERSION=2.6"]

	def builderName(self):return 'buildbot-{0}'.format(self.__class__.__name__)

	def slaves(self):
		slaves=[]
		for (k,v) in self.pips_archs.iteritems():
			slaves.append(BuildSlave(k,v))
		return slaves

	def schedulers(self):
		schedulers=[]
		for arch in self.pips_archs.iterkeys():
			if self.get_factory(arch):
				scheduler=Nightly(
					'nightly-{0}-{1}'.format(self.builderName(),arch),
					[ '{0}-{1}'.format(self.builderName(),arch) ],
					hour=self.hour,
					minute=self.minute
				)
				schedulers.append(scheduler)
		return schedulers

	def builders(self):
		builders=[]
		for arch in self.pips_archs.iterkeys():
			builder= {
				'name':'{0}-{1}'.format(self.builderName(),arch),
				'slavename':arch,
				'builddir': '{0}/{1}'.format(arch,self.builderName()),
				'factory':self.get_factory(arch)
			}
			# filter out undesired builders
			if builder['factory']: 
				builders.append(builder)
		return builders

	def get_factory(self,arch):
		return self.initFactory(arch)

class Polylib(Pipsbuild):
	hour=3
	minute=50
	def initFactory(self,arch):
		return GNUAutoconf(Git("http://repo.or.cz/r/polylib.git"),
			test=None,
			distcheck=None,
			configureFlags=self.configure_flags)

class Newgen(Pipsbuild):
	hour=3
	minute=50
	targz = "newgen-0.1.tar.gz"
	def initFactory(self,arch):
		f= GNUAutoconf(SVN("http://svn.cri.ensmp.fr/svn/newgen/trunk"),
			test=None,
			configureFlags=self.configure_flags)
		if arch == "linux-64":
			f.addStep(FileUpload(slavesrc=os.path.join("BUILD",self.targz),
                        	masterdest=self.www+self.targz, mode=0644))
		return f


class Linear(Pipsbuild):
	hour=4
	minute=0
	targz = "linear-0.1.tar.gz"
	def initFactory(self,arch):
		f = GNUAutoconf(SVN("http://svn.cri.ensmp.fr/svn/linear/trunk"),
			test=None,
			configureFlags=self.configure_flags)
		if arch == "linux-64":
			f.addStep(FileUpload(slavesrc=os.path.join("BUILD",self.targz),
						masterdest=self.www+self.targz, mode=0644))
		return f

class Pips(Pipsbuild):
	hour=4
	minute=20
	targz = "pips-0.1.tar.gz"
	def __init__(self):
		super(Pips,self).__init__()
		for l in [ self.configure_flags, self.dist_configure_flags ]:
			l.append("--enable-pyps")
			l.append("--enable-pyps-extra")
			l.append("--enable-fortran95")
			l.append("--enable-gpips")
			#l.append("--enable-hpfc")

	def initFactory(self,arch):
		self.make = "gmake" if arch == "bsd-64" else "make"
		f = GNUAutoconf( SVN("http://svn.cri.ensmp.fr/svn/pips/trunk"),
			configureFlags=self.configure_flags,
			compile=[self.make, "all"],
			test=None,
			install=[self.make, "install"],
			distcheck=None #[self.make, "distcheck"]
			)
		if arch == "linux-64":
			f.addStep(FileUpload(slavesrc=os.path.join("BUILD",self.targz),
						masterdest=self.www+self.targz, mode=0644))
		return f

class PipsValidate(Pipsbuild):
	hour=4
	minute=50

	def initFactory(self,arch):
		if arch != "linux-64":
			self.make="gmake"
		else:
			self.make="make"
		f = factory.BuildFactory()
		f.addStep(SVN("http://svn.cri.ensmp.fr/svn/validation/trunk"))
		s=PipsValid(command=[self.make, 'validate',
		#		'TARGET=TutorialPPoPP2010',
			],
			logfiles={'summary':'SUMMARY.short'},
			env={
				'PATH':'{0}:{1}'.format(os.environ['PATH'],os.path.join(self.install_prefix,"bin")),
				'PYTHONPATH':'{0}'.format(os.path.join(self.install_prefix,"lib","python2.6","site-packages","pips")),
				'LD_LIBRARY_PATH':'{0}'.format(os.path.join(self.install_prefix,"lib")),
				'PIPS_F77':'gfortran',
				'PIPS_F90':'gfortran',
				})
		f.addStep(s)
		return f

class PipsGet(Pipsbuild):
	hour=7
	minute=0

	def initFactory(self,arch):
		if arch != "linux-64": return None
		f = factory.BuildFactory()
		f.addStep(FileDownload(mastersrc="/var/www/pips/get-pips4u.sh",
                            slavedest="get-pips4u.sh"))
		f.addStep(Cleaning(command=["rm","-rf","/tmp/pipsget"]))
		f.addStep(GetPipsTarball(
			command=["sh","get-pips4u.sh","--devel",
				"--srcdir", "/tmp/pipsget/src",
				"--prefix", "/tmp/pipsget/root",
				"--force",
				"--debug"])
			)
		f.addStep(Cleaning(command=["rm","-rf","/tmp/pipsget"]))
		f.addStep(GetPipsSvn(
			command=["sh","get-pips4u.sh",
				"--srcdir", "/tmp/pipsget/src",
				"--prefix", "/tmp/pipsget/root",
				"--force",
				"--debug"])
			)
		return f
				


