import workspace_gettime

def getAllLoops(entity):
	loops = []
	for x in entity.loops():
		loops = loops + [x] + getAllLoops(x)
	return loops

class workspace:
	detach_times = 0
	def __init__(self, ws, sources, *args, **kwargs):
		with open("log", 'w') as f:
			pass
			
		self.elems = []
		self.runCount = 0
		self.loopCount = -1
		self.ws = ws
		
		#PIPS has some problems with some "dummy parameter __x" so we moreve the math during the analysis
		for file in sources:
			with open(file, 'r') as f:
				read_data = f.read()
				#Don't put the include more than once
				if read_data.find('//#include <math.h>') != -1:
					continue
			with open(file, 'w') as f:
				read_data = read_data.replace("#include <math.h>", "//#include <math.h>")
				f.write(read_data)
	
	def pre_goingToRunWith(self, files, outdir):
		for file in files:
			with open(file, 'r') as f:
				read_data = f.read()
				#Don't put the include more than once
				if read_data.find('//#include <math.h>') == -1:
					continue
			with open(file, 'w') as f:
				read_data = read_data.replace("//#include <math.h>", "#include <math.h>")
				f.write(read_data)
	
	def detach(self):
		if len(self.elems) is 0:
			return

		copy = self.elems
		self.elems = []
		
		for x in copy:
			workspace.detach_times = workspace.detach_times + 1
			x.run(self.ws)
			self.runCount = self.runCount + 1
		
		print ("DETACHED === {0} ===".format(workspace.detach_times))
	
	def pre_set_property(self, **props):
		self.detach()
		
	def pre___iter__(self):
		self.detach()
	
	def pre___getitem__(self, name):
		self.detach()
		
	def pre___setitem__(self, i):
		self.detach()
		
	def pre__build_module_list(self):
		self.detach()
	
	def pre_filter(self, *args, **kwargs):
		self.detach()
	
	def pre_activate(self, phase):
		self.detach()
		
	def pre_compile(self, *args, **kwargs):
		self.detach()
		
	def pre_save(self, *args, **kwargs):
		self.detach()
	
	def run (self, elem):
		self.elems.append(elem)
		
	def getAllLoops(self):
		self.detach()
		if self.loopCount == self.runCount:
			return self.allLoops;
		"""Returns all the loops and subloops of an entity"""
		self.allLoops = getAllLoops(self.ws.filter())
		self.loopCount = self.runCount
		return self.allLoops
			
class transfo:
	"""stores informations concerning a code transformation"""
	def __init__(self,module,transfo, loop=None, **props):
		self.transfo=transfo
		self.modname = module
		self.loop = loop
		self.props = props
	def __str__(self):return "".join([str(property(prop, val)) for (prop, val) in self.props.items()]) + "transformation:"+self.transfo+" on module " + self.modname + "\n"
	def run(self,wsp):
		"""run this transformation on module `name'"""
		print ("running transformation " + self.transfo)
		#getAllLoops(wsp[self.modname])
		wsp.getAllLoops()
		with open("log", 'a') as f:
			print >> f, "apply " + self.transfo + "[" + str(self.modname) + "]"
		#wsp[self.modname].apply(self.transfo)
		if not self.loop:
			getattr(wsp[self.modname], self.transfo.lower())(**self.props)
		else:
			getattr(wsp[self.modname].loops(self.loop), self.transfo.lower())(**self.props)
	def __cmp__(self,other):
		if type(other).__name__ != 'transfo': return -1
		n = cmp(self.modname, other.modname)
		if n != 0:
			return n
		n = cmp(self.loop, other.loop)
		if n != 0:
			return n
		n = cmp(self.props, other.props)
		if n != 0:
			return n
		return cmp(self.transfo,other.transfo)
	def __hash__(self):
		return hash("{0}:{1}:{2}:{3}".format(self.transfo,self.modname,self.loop,self.props))

class property:
	"""stores informations concerning a pips property"""
	def __init__(self,prop, value):
		self.prop=prop.upper()
		self.val=value
	def __str__(self):return "property:{0} value:{1}\n".format(self.prop, self.val)
	def run(self, workspace):
		"""set the property on current workspace"""
		print ("setting property " + self.prop + " to " + str(self.val))
		with open("log", 'a') as f:
			print >> f, "setproperty " + self.prop + " " + str(self.val)
		
		workspace.set_property(** {self.prop:self.val} )
	def __cmp__(self,other):
		if type(other).__name__ != 'property': return -1
		n=cmp(self.prop,other.prop) 
		if n == 0: return cmp(self.val,other.val) 
		else: return n
	def __hash__(self):
		return hash("{0}:{1}".format(self.prop,self.val))

