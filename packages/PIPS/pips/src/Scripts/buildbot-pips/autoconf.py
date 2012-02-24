# Copyright (c) 2012, Serge Guelton <serge.guelton@telecom-bretagne.eu>
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
#     Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#     Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

from buildbot.process.factory import BuildFactory

from buildbot.steps.source import SVN
from buildbot.steps.shell import ShellCommand, Configure, Compile

from buildbot.status.builder import SUCCESS

class Bootstrap(ShellCommand):
	""" Bootstraps the autotool environnment. """
	name = "bootsrap"
	haltOnFailure = 1
	description = ["bootstraping"]
	descriptionDone = ["bootstrap"]
	command = ["autoreconf","-vi"]
	
class Distcheck(ShellCommand):
	"""Distribution checking step"""
	name = "distcheck"
	haltOnFailure = 1
	description = ["checking distribution"]
	descriptionDone = ["check distribution"]

class Install(ShellCommand):
	"""Installation step"""
	name = "install"
	haltOnFailure = 1
	description = ["installing"]
	descriptionDone = ["install"]

class Cleaning(ShellCommand):
	"""Failsafe cleaning step"""
	name = "cleaning"
	haltOnFailure = 0
	description = ["cleaning repository"]
	descriptionDone = ["clean repository"]
	def evaluateCommand(self,cmd): return SUCCESS
	

class GNUAutoconf(BuildFactory):
	"""Enhanced version of builbot's GNUAutoconf factory.
           Makes it possible to add distcheck and boostrap steps,
	   and uses a builddir instead of building in the sources."""

	build_dir = "build/_build"

    	def __init__(self, source,
		 	bootstrap=["autoreconf","-vi"],
		 	configure=["../configure"],
                 	configureEnv={},
                 	configureFlags=[],
                 	compile=["make", "all"],
                 	test=["make", "check"],
			install=["make", "install"],
		 	distcheck=["make", "distcheck"],
			clean=True):
		"""Perform the following steps:
		   source, bootstrap, configure, compile, test, install, distcheck
		   any step can be disabled by setting the appropriate argument to `None'
		   except source and boostrap, all steps are done in `build_dir'
		   distcheck is made using configureFlags, stripping `--prefix' out
		   set clean to False to disable deletion of `build_dir' before configure"""
		BuildFactory.__init__(self, [source])
		if bootstrap: # some people also call it autogen
			self.addStep(Bootstrap, command=bootstrap) 
		if clean:
			self.addStep(Cleaning, command=["rm", "-rf", GNUAutoconf.build_dir])
		if configure:
			self.addStep(Configure, command=configure+configureFlags, env=configureEnv, workdir=GNUAutoconf.build_dir)
		if compile:
			self.addStep(Compile, command=compile, workdir=GNUAutoconf.build_dir)
		if test:
			self.addStep(Test, command=test, workdir=GNUAutoconf.build_dir)
		if install:
			self.addStep(Install, command=install, workdir=GNUAutoconf.build_dir)
		if distcheck:
			distcheck_configure_flags=[ flag for flag in configureFlags if not flag.startswith("--prefix")]
			self.addStep(Distcheck, command=distcheck+["DISTCHECK_CONFIGURE_FLAGS="+" ".join(distcheck_configure_flags)], workdir=GNUAutoconf.build_dir)
		
