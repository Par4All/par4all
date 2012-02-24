from buildbot.steps.shell import ShellCommand,Configure
from buildbot.status.builder import SUCCESS, WARNINGS, FAILURE

class PipsValid(ShellCommand):
	name = "pips validation"
	haltOnFailure = 0
	description = ["running validation"]
	descriptionDone = ["run validation"]

	def commandComplete(self,cmd):
		fulltxt=""
		failure=0
		changed=0
		for line in cmd.logs['summary'].readlines():
			if line.startswith("failed"):failure+=1
			elif line.startswith("changed"):changed+=1
			fulltxt+=line
		self.descriptionDone=["changed/failed: {0}/{1}".format(changed,failure)]

	def evaluateCommand(self,cmd):
		for line in cmd.logs['summary'].readlines():
			if line.startswith("failed"):return FAILURE
		return SUCCESS

class GetPipsTarball(ShellCommand):
	name = "get-pips4u.sh"
	description = ["testing get-pips4u.sh"]
	descriptionDone = ["test get-pips4u.sh"]
	
class GetPipsSvn(ShellCommand):
	name = "get-pips4u.sh --devel"
	description = ["testing get-pips4u.sh --devel"]
	descriptionDone = ["test get-pips4u.sh --devel"]
