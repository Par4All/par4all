import pypips
import os
import random
from string import split, upper
import shutil 

modules = {}

module_ext = ""

def workspace_dir():return workspace()+".database/"


class module:
	def __init__(self,name):
		self.name=name
		modules[name]=self

	def show(self,rc):
		return split(pypips.show(upper(rc),self.name))[-1]

	def apply(self,phase):
		pypips.apply(upper(phase),self.name)

	def display(self,rc="printed_file"):
		return pypips.display(upper(rc),self.name)

	def code(self):
		self.apply("print_code")
		rcfile=self.show("printed_file")
		return file(workspace_dir()+rcfile).readlines()

	def save(self,dest=""):
		if not dest:
			dest=self.name + module_ext
		self.apply("unsplit")
		src=workspace_dir()+"Src/"+self.name+module_ext
		if not os.path.exists(src):
			src=workspace_dir()+self.show("printed_file")
		shutil.copy(src,dest)

	def inline(self):self.apply("inlining")
### helpers /!\ do not touch this line /!\


def info(topic):
	return split(pypips.info(topic))

def workspace():
	return (info("workspace")[0])


def set_property(prop,value):
	if type(value) is bool:
		val=upper(str(value))
	elif type(value) is str:
		def stringify(s): return '"'+s+'"'
		val=stringify(value)
	else:
		val=str(value)
	pypips.set_property(upper(prop),val)


def create(*sources):
	global module_ext
	workspace="PYPS"+str(random.Random().randint(0,100000))
	pypips.create(workspace, list(sources))
	if os.path.splitext(sources[0])[1] == ".c":
		module_ext=".c"
		pypips.activate("C_PARSER");
		set_property("PRETTYPRINT_C_CODE",True)
		set_property("PRETTYPRINT_STATEMENT_NUMBER", False)
	else:
		module_ext=".f"
	for m in info("modules"):
		modules[m]=module(m)

def close():
	ws=workspace()
	pypips.quit()
	pypips.delete_workspace(ws)





