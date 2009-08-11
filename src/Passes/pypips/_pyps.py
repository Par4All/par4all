# coding=iso-8859-15
import pypips
import os
from string import split, upper, join
import shutil 




class module:
	def __init__(self,ws,name,source=""):
		self.name=name
		self.source=source
		self.ws=ws

	def show(self,rc):
		return split(pypips.show(upper(rc),self.name))[-1]

	def apply(self,phase):
		pypips.apply(upper(phase),self.name)

	def display(self,rc="printed_file"):
		return pypips.display(upper(rc),self.name)

	def code(self):
		self.apply("print_code")
		rcfile=self.show("printed_file")
		return file(self.ws.dir()+rcfile).readlines()

	def __update_props(self,passe,props):
		""" change a property dictionnay by appending the passe name to the property when needed """
		for name,val in props.iteritems():
			if upper(name) not in self.all_properties:
				del props[upper(name)]
				props[upper(passe+"_"+name)]=val
				#print "warning, changing ", name, "into", passe+"_"+name
		return props

### helpers /!\ do not touch this line /!\

class workspace:

	def initialize(self,sources2):
		workspace=os.path.basename(os.tempnam("","PYPS"))
		def helper(x,y):
			if type(y).__name__ == 'list':return x+y
			else: return x+[y]
		sources=reduce(helper,sources2,[])
		pypips.create(workspace, sources)
		self.modules = {}
		if os.path.splitext(sources[0])[1] == ".c":
			self.module_ext=".c"
			pypips.activate("C_PARSER");
			self.set_property(PRETTYPRINT_C_CODE=True,PRETTYPRINT_STATEMENT_NUMBER=False)
		else:
			self.module_ext=".f"
		for m in self.info("modules"):
			self.modules[m]=module(self,m,sources[0])
		self.name=self.info("workspace")[0]
		self.cleared=False

	def __init__(self,*sources2):
		self.initialize(sources2)

	def __getitem__(self,i):
		return self.modules[i]

	def __setitem__(self,i):
		return self.modules[i]

	def info(self,topic):
		return split(pypips.info(topic))

	def dir(self):return self.name+".database/"

	def _set_property(self,props):
		for prop,value in props.iteritems():
			if type(value) is bool:
				val=upper(str(value))
			elif type(value) is str:
				def stringify(s): return '"'+s+'"'
				val=stringify(value)
			else:
				val=str(value)
			pypips.set_property(upper(prop),val)

	def set_property(self,**props):
		self._set_property(props)

	def save(self,indir="",with_prefix=""):
		pypips.apply("UNSPLIT","%ALL")
		saved=[]
		if indir:
			if not os.path.exists(indir):
				os.makedirs(indir)
			if not os.path.isdir(indir): raise ValueError("'" + indir + "' is not a directory") 
			for s in os.listdir(self.dir()+"Src"):
				cp=indir+"/"+s
				shutil.copy(self.dir()+"Src/"+s,cp)
				saved+=[cp]
		else:
			for s in os.listdir(self.dir()+"Src"):
				cp=with_prefix+s
				shutil.copy(self.dir()+"Src/"+s,cp)
				saved+=[cp]
		return saved

	def compile(self,CC="gcc",CFLAGS="-O2 -g", LDFLAGS="", link=True, outdir=".", outfile="",extrafiles=[]):
		if not os.path.isdir(outdir): raise ValueError("'" + indir + "' is not a directory") 
		otmpfiles=self.save(indir=outdir)+extrafiles
		command=[CC,CFLAGS]
		if link:
			if not outfile:
				outfile=self.name
			command+=otmpfiles
			command+=[LDFLAGS]
			command+=["-o", outfile]
		else:
			command+=["-c"]
			command+=otmpfiles
		#print "running", " ".join(command)
		if os.system(" ".join(command)):
			if not link: map(os.remove,otmpfiles)
		return outfile

	def activate(self,phase):
		pypips.activate(phase)


	def quit(self):
		self.cleared=True
		pypips.quit()
		pypips.delete_workspace(self.name)

	def __del__(self):
		if not self.cleared:self.quit()

