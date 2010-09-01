#!/usr/bin/env python
"""
This module provide three ways of exploring the transformation space for a given module in a given programm
- brute force exploration
- greedy exploration
- genetic algorithm based exploration
"""
#import pdb
import workspace_gettime
import random
from optparse import OptionParser, Option
import operator
import sys
import subprocess
import os
import shutil
import pyrops
import ConfigParser

from string import upper

#
##
#

class Transfo:
	"""stores informations concerning a code transformation"""
	def __init__(self,module,transfo, loop=None, **props):
		self.transfo=transfo
		self.modname = module
		self.loop = loop
		self.props = props
	def __str__(self):return "".join([str(Property(prop, val)) for (prop, val) in self.props.items()]) + "transformation:"+self.transfo+" on module " + self.modname + "\n"
	def run(self,wsp):
		"""run this transformation on module `name'"""
		if wsp.verbose:
			print ("running transformation " + self.transfo)
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

class Property:
	"""stores informations concerning a pips property"""
	def __init__(self,prop, value):
		self.prop=prop.upper()
		self.val=value

	def __str__(self):return "property:{0} value:{1}\n".format(self.prop, self.val)

	def run(self, workspace):
		"""set the property on current workspace"""
		if workspace.verbose:print ("setting property " + self.prop + " to " + str(self.val))
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

class Gene:
	"""a gene contains a transformation with associated properties
	it represents a parametrized transformation on a module"""
	def __init__(self,*codons):
		self.codons=codons
		
	def run(self, wsp):
		"""apply all the transformation in the gene on module `name'"""
		[ x.run(wsp) for x in self.codons ] 

	def __str__(self):
		return reduce(lambda x,y:x+str(y),self.codons,"")

	def __cmp__(self,other):
		n = cmp(len(self.codons),len(other.codons))
		if n != 0: return n
		else:
			for i in range(0,len(self.codons)):
				n = cmp(self.codons[i],other.codons[i])
				if n != 0: return n
			return 0
#
##
#

def called(self):
	callers=self.callers()
	while callers:
		caller=callers.pop()
		if caller.name == "main": return True
		else: callers+=caller.callers()
	return self.name == "main"


	
class Generator(object):
	def __init__(self):
		self.generated = {}
	
	def fastGenerating(self, g):
		if g in self.generated:
			return self.generated[g]
		return None

class UnrollGenerator(Generator):
	"""Generates an unroll transformation"""
	def generate(self, worksp):
		genes = []
		for m in worksp.fun:
			if called(m):
				loops=m.loops()
				while loops:
					loop = loops.pop()
					loops+=loop.loops()
					for r in [2,4,8]:
						genes = genes + [Gene(Transfo(m.name, "UNROLL", loop=loop.label, unroll_rate=r))]
		return genes

class FusionGenerator(Generator):
	"""Generates a loop fusion transformation"""
	def generate(self, worksp):
		genes = []
		for m in worksp.fun:
			if called(m):
				for l in m.loops()[:-1]:
					genes = genes + [Gene(Transfo(m.name, "FORCE_LOOP_FUSION", loop=l.label))]
		return genes


class RedVarExpGenerator(Generator):
	"""Generates a partialeval transformation"""
	def generate(self, worksp):
		
		loops = worksp.getAllLoops()
		
		genes = []
		for loop in loops:
			genes = genes + [Gene(Transfo(loop.module.name, "REDUCTION_VARIABLE_EXPANSION", loop = loop.label))]
		return genes

class InterchangeGenerator(Generator):
	"""Generates a loop interchange transformation"""
	def generate(self, worksp):
		
		genes = []
		
		loops = worksp.getAllLoops()
		
		for loop in loops:
			if len(loop.loops()) > 0:
				genes = genes + [Gene(Transfo(loop.module.name, "LOOP_INTERCHANGE", loop=loop.label))]
		return genes

class DummyGenerator(Generator):
	"""Generates a partialeval transformation"""
	def __init__(self, name, **args) :
		self.name = name
		self.properties = args
		super(DummyGenerator, self).__init__()
	
	def generate(self, worksp):
		genes = []
		for module in worksp.fun:
			if called(module):
				genes.append( Gene (Transfo(module.name, self.name, **self.properties)))
			
		return genes

class UnfoldGenerator(Generator):
	def generate(self, worksp):
		""" Otherwise let's lookup everything. """
		genes = []
		for module in worksp.fun:
			if module.callees() and called(module):
				genes = genes + [Gene(Transfo(module.name, "UNFOLDING"))]
		return genes

class InlineGenerator(Generator):
	def generate(self, worksp):
		""" Otherwise let's lookup everything. """
		genes = []
		for module in worksp:
			if called(module):
				for caller in module.callers():
					genes = genes + [Gene(Transfo(module.name, "INLINING", inlining_purge_labels=True,
						inlining_callers=caller.name))]
		
		return genes

#
##
#
	
#
##
#
def getwdir(sources):
	return "WDIR_"+"".join("-".join(sources).split('/'))

	
class Individual(object):
	def __init__(self,args):
		self.args=args
		self.genes=[]
		self.execution_time=0
		self.min_time = 0
		self.max_time = 0

		self.foutname="_".join([str(random.randint(0,1000000)) , os.path.basename(args.sources[0])])
		# create workspace
		self.ws= pyrops.pworkspace(self.args.sources, verbose=args.verbose,recoverInclude = True, parents = [workspace_gettime.workspace])
		self.ws.activate("MUST_REGIONS")
		self.ws.activate("PRECONDITIONS_INTER_FULL")
		self.ws.activate("TRANSFORMERS_INTER_FULL")
		self.ws.activate("RICE_SEMANTICS_DEPENDENCE_GRAPH")
		self.ws.activate("RICE_REGIONS_DEPENDENCE_GRAPH")
		self.ws.activate("REGION_CHAINS")

		self.ws.set_property(RICEDG_STATISTICS_ALL_ARRAYS=True)
		self.ws.set_property(C89_CODE_GENERATION=True)
		self.ws.set_property(CONSTANT_PATH_EFFECTS=False)

	def push(self,gene):
		try:
			gene.run(self.ws)
			self.genes.append(gene)
		except:pass

	def rate(self):
		if not self.execution_time :
			wdir=os.sep.join([getwdir(self.args.sources),self.foutname])
			self.ws.save(indir=wdir)
			cflags=os.environ["PIPS_CPP_FLAGS"]+" -I. -O0 -w"

			#random names in case several processes are running at the same time
			randString = str(random.randint(0, 1000000)) + str(random.randint(0, 1000000))
			exec_out = self.ws.compile(CC="gcc", CFLAGS=cflags,outdir=wdir,outfile="/tmp/" + randString + ".out")
			elapsed=[]

			#print "running",exec_out
			for i in range(0,self.args.get_time):
				t=-1
				while t<0:
					subprocess.Popen([exec_out] + self.args.bench_args,stdout=file("/dev/null"),stderr=subprocess.STDOUT).wait()
					with open("_pyps_time.tmp", "r") as f:
						t = f.readline()
					#print "measured:" , t
				elapsed+=[int(t)]
			elapsed.sort()
			self.execution_time=elapsed[len(elapsed)/2]
			self.min_time = elapsed[0]
			self.max_time = elapsed[-1]

	def rip(self):
		self.ws.close()
	
	def __str__(self):
		s=reduce(lambda x,y:x+" " +y,self.args.sources,"sources:")
		s+= " out:"+self.foutname+"\n"
		s+= "workspace:"+self.ws.name+"\n"
		s+= "execution time:" + str(self.execution_time) + " (min: {0}, max: {1})\n".format(self.min_time, self.max_time)
		for t in self.genes:
			s+=str(t)
		return s

	
#
##
#

class Algo(object):
	def __init__(self,args):
		self.nbgen=args.gens
		self.args=args
		self.pool=[Individual(args)] # we start with a dummy individual
	
	def run(self):
		# process nbgen generation
		#pdb.set_trace()
		while self.nbgen > 0:
			self.nbgen-=1
			self.populate()
			self.rate()
			self.select()
		self.sort()
		for individual in self.pool:individual.rip()
		return self.pool

	def rate(self):
		for individual in self.pool:
			individual.rate()

	def sort(self):
		self.pool.sort(key=operator.attrgetter('execution_time'))

class Greedy(Algo):
	def __init__(self,args):
		super(Greedy,self).__init__(args)

	def populate(self):
		newpool=[]
		for individual in self.pool:
			geneCandidates=[]
			for generator in self.args.generators:
				geneCandidates+=generator.generate(individual.ws)
			for gene in geneCandidates:
				newindividual=Individual(self.args)
				for igene in individual.genes:
					newindividual.push(igene)
				newindividual.push(gene)
				newpool.append(newindividual)
		self.pool+=newpool

	def select(self):
		self.sort()
		for individual in self.pool[self.args.popsize:]:
			individual.rip()
		self.pool=self.pool[0:self.args.popsize]

#
##
#

class Genetic(Algo):
	def __init__(self,args):
		super(Genetic,self).__init__(args)
		# init population
		for i in range(1,args.popsize):
			self.pool.append(Individual(self.args))

	def populate(self):
		newpool=[]
		for individual in self.pool:
			geneCandidates=[]
			for generator in self.args.generators:
				geneCandidates+=generator.generate(individual.ws)
			# add a random gene among the existing one
			[ gene ] = random.sample(geneCandidates,1)
			newindividual=Individual(self.args)
			for igene in individual.genes:
				newindividual.push(igene)
			newindividual.push(gene)
			newpool.append(newindividual)
		self.pool+=newpool

	def select(self):
		# ensure we have an even population
		if len(self.pool) % 2  :
			raise Exception("not even")
		half_individual=set(random.sample(self.pool,len(self.pool)/2))
		other_half_individual=set(self.pool).difference(half_individual)
		def select_individual(i0,i1):
			if i0.execution_time < i1.execution_time:
				if self.args.verbose: print i0,"<<<<<<<<<<",i1
				i1.rip()
				return i0
			else:
				if self.args.verbose: print i0,">>>>>>>>>>",i1
				i0.rip()
				return i1
		self.pool=map(select_individual,half_individual,other_half_individual)
	
#
##
#

def ParseConfigFile(args):
	""" returns a set of generator """

	args.generators = []
	parser = ConfigParser.RawConfigParser()
	parser.read("pypsearch.cfg")
	
	base = {"inline":InlineGenerator(),
			"unroll":UnrollGenerator(),
			"unfold":UnfoldGenerator(),
			"fusion":FusionGenerator(),
			"red_var_exp":RedVarExpGenerator(),
			"interchange":InterchangeGenerator()
			}
	
	for x in parser.items("generators"):
		if x[0] not in base:
			print "Error when parsing config file, as " + x[0] + " is not a generator"
		else:
			args.generators += [base[x[0]]]
	
	for x in parser.items("basicGenerators"):
		props = x[1].split(',')
		propvals = {}
		
		for propval in props:
			splitProp = propval.split('=')
			if len(splitProp) != 2 or len(splitProp[1]) == 0:
				continue
			prop = splitProp[0]
			val = splitProp[1]
			if val[0] == '"':
				val = val[1:-1]
			elif upper(val) == "TRUE" or upper(val) == "FALSE":
				val = bool(val)
			else:
				val = int(val)
			propvals[prop] = val
		args.generators.append(DummyGenerator(x[0], **propvals))
	
	if not args.generators :
		print "No generator given in the config file (pypsearch.cfg), so using only the inline generator"
		args.generators = [base["inline"]]
	
	print "Using generators:",
	for g in args.generators:
		try:print "%s(%s)" % ( g.__class__.__name__, g.name ) ,
		except AttributeError:print g.__class__.__name__,
	print



def ParseCommandLine():
	parser = OptionParser(description="Pypsearch - Automated exploration of the set of transformations with python.",
						  usage="usage: %prog [options] sourcename")
	parser.add_option('--algo', default="genetic",type=str, help='search algorithm to use')
	parser.add_option('--log', help='log file to save the best results')
	parser.add_option('--gens', default=1, type=int, help='Number of generations for the genetic algorithm')
	parser.add_option('--crossovers', default=1, type=int, help='Number of crossovers to perform each generation')
	parser.add_option('--tournaments', default=3, type=int, help='Number of winners of a tournament for the genetic algorithm')
	parser.add_option('--popsize', type=int, default=1,help='Number of individuals for the genetic algorithm')
	parser.add_option('--CPPFLAGS', default='', help='Optional added arguments to the compiler')
	parser.add_option('--bench-args',default='',help='Optionnal arguments for benchmarking code',dest='bench_args')
	parser.add_option('--bench-iter', default=20,type=int, help='Number of iteration for benchmarking',dest="get_time")
	parser.add_option('-v', action="store_true", help='be very talkative',dest="verbose")
	
	(args, files) = parser.parse_args()
	args.sources = files
	# verify some settings
	if not args.sources:
		print ("Error, you need to input at least one source file")
		sys.exit(2)
	available_algos = { "genetic": Genetic , "greedy": Greedy }
	args.algo=available_algos[args.algo]
	print "Using %s as search algorithm" % args.algo.__name__

	if args.bench_args:
		args.bench_args=args.bench_args.split(' ')
	else:
		args.bench_args=[]
		
	if args.log == None:
		args.log = sys.stdout
	else:
		args.log = open(args.log, 'w')

	#random.start_saving("log_random")
	
	os.environ["PIPS_CPP_FLAGS"]=args.CPPFLAGS

	ParseConfigFile(args)
	return args

def pypsearch():
	args=ParseCommandLine()
	
	# init 
	print "Create xp directory"
	workspacedir = getwdir(args.sources)
	if os.path.exists(workspacedir):
		shutil.rmtree(workspacedir)
	os.mkdir(workspacedir)
	[ shutil.copy(x,workspacedir) for x in args.sources	]
	
	#
	# launch algo
	results=[]
	print "Running algorithm"
	results=args.algo(args).run()
	
	print >> args.log , "-- best results" 
	
	original =  Individual(args)
	original.rate()
	original.rip()
	
	for r in results:
		print >> args.log, r
		print >> args.log, "--"
	print ("-- original result --")
	print original
	print ("done !")
	#random.finish_saving()

	

if __name__ == "__main__":
	pypsearch()
