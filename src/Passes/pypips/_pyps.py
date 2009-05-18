# coding=iso-8859-15
import pypips
import os
from string import split, upper, join
import shutil 

modules = {}
module_ext = ""

def workspace_dir():return workspace()+".database/"


class module:
	def __init__(self,name, source=""):
		self.name=name
		self.source=source
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
		if self.source:
			src=workspace_dir()+"Src/"+self.source
			if not os.path.exists(src):
				#print "unsplitted file not found, saving module file"
				src=workspace_dir()+self.show("printed_file")
			shutil.copy(src,dest)
		else:
			print "not saving : no source available"

	def compile(self,CC="gcc",CFLAGS="-O2 -g", LDFLAGS="", link=True, outfile=""):
		tmpfile=os.tempnam()
		otmpfile=tmpfile+".c"
		file(tmpfile,"w").close()
		self.save(otmpfile)
		command=[CC,CFLAGS]
		
		if link:
			if not outfile:
				outfile=self.name
			command+=[otmpfile,LDFLAGS]
		else:
			if not outfile:
				outfile=self.name+".o"
			command+=["-c" , otmpfile]
		command+=["-o", outfile]
		#print "running", " ".join(command)
		if os.system(" ".join(command)):
			os.remove(otmpfile)
			os.remove(tmpfile)
		return outfile

	def _update_props(self,passe,props):
		""" change a property dictionnay by appending the passe name to the property when needed """
		for name,val in props.iteritems():
			if upper(name) not in self.all_properties:
				del props[upper(name)]
				props[upper(passe+"_"+name)]=val
				#print "warning, changing ", name, "into", passe+"_"+name
		return props

### helpers /!\ do not touch this line /!\


def info(topic):
	return split(pypips.info(topic))

def workspace():
	return (info("workspace")[0])

def _set_properties(props):
	for prop,value in props.iteritems():
		if type(value) is bool:
			val=upper(str(value))
		elif type(value) is str:
			def stringify(s): return '"'+s+'"'
			val=stringify(value)
		else:
			val=str(value)
		pypips.set_property(upper(prop),val)

def set_property(**props):
	_set_properties(props)


def create(*sources):
	global module_ext
	workspace=os.path.basename(os.tempnam("","PYPS"))
	pypips.create(workspace, list(sources))
	if os.path.splitext(sources[0])[1] == ".c":
		module_ext=".c"
		pypips.activate("C_PARSER");
		set_property(PRETTYPRINT_C_CODE=True,PRETTYPRINT_STATEMENT_NUMBER=False)
	else:
		module_ext=".f"
	#if len(sources) > 1:
	#	print "/!\\ the save method will not work correctly /!\\"
	for m in info("modules"):
		modules[m]=module(m,sources[0])

def close():
	ws=workspace()
	pypips.quit()
	pypips.delete_workspace(ws)





