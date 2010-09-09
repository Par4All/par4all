#!/usr/bin/env python
"""
This module provide three ways of exploring the transformation space for a given module in a given programm
- brute force exploration
- greedy exploration
- genetic algorithm based exploration
"""

import os

import tempfile
import pdb
import workspace_gettime
import random
from optparse import OptionParser, Option
import operator
from operator import xor
import sys
import subprocess
import shutil
import ConfigParser
from string import upper

# must set this first
os.environ['PYRO_PORT_RANGE']='1000'
import pyrops


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

	def __str__(self):
		s= "".join([str(Property(prop, val)) for (prop, val) in self.props.items()]) +\
				"transformation:"+self.transfo+" on module " + self.modname + "\n"
		if self.loop:s+= "with loop:" + self.loop + "\n"
		return s

	def run(self,wsp):
		"""run this transformation on module `name'"""
		if self.loop:
			getattr(wsp[self.modname].loops(self.loop), self.transfo.lower())(**self.props)
		else:
			getattr(wsp[self.modname], self.transfo.lower())(**self.props)

	def __hash__(self):
		return hash("{0}:{1}:{2}:{3}".format(self.transfo,self.modname,self.loop,self.props))

	def __cmp__(self,other):
		return cmp(self.__hash__(),other.__hash__())

class Property:
	"""stores informations concerning a pips property"""
	def __init__(self,prop, value):
		self.prop=prop.upper()
		self.val=value

	def __str__(self):return "property:{0} value:{1}\n".format(self.prop, self.val)

	def run(self, workspace):
		"""set the property on current workspace"""
		if workspace.verbose:print ("setting property " + self.prop + " to " + str(self.val))
		workspace.set_property(** {self.prop:self.val} )

	def __hash__(self):
		return hash("{0}:{1}".format(self.prop,self.val))

	def __cmp__(self,other):
		return cmp(self.__hash__(),other.__hash__())

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

	def __hash__(self):
		return reduce(xor,[ x.__hash__() for x in self.codons ])

	def __cmp__(self,other):
		return cmp(self.__hash__(),other.__hash__())
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
	pass

class DummyGenerator(Generator):
	"""Generates a transformation without constraints"""
	def __init__(self, name, **args) :
		self.name = name
		self.properties = args
	
	def generate(self, individual):
		genes = []
		for module in individual.ws.fun:
			if called(module):
				genes.append( Gene (Transfo(module.name, self.name, **self.properties)))
		return genes

class UniqGenerator(DummyGenerator):
	"""Generates a transformation that was not just applied"""
	def generate(self, individual):
		genes = []
		for module in individual.ws.fun:
			if called(module):
				gene =Gene (Transfo(module.name, self.name, **self.properties))
				if not individual.genes or gene != individual.genes[-1]:
					genes.append( gene )
		return genes

class UnrollGenerator(Generator):
	"""Generates an unroll transformation"""
	def generate(self, individual):
		genes = []
		for m in individual.ws.fun:
			if called(m):
				loops=m.loops()
				while loops:
					loop = loops.pop()
					loops+=loop.loops()
					for r in [2,4,8]:
						genes.append(Gene(Transfo(m.name, "UNROLL", loop=loop.label, unroll_rate=r)))
		return genes

class FusionGenerator(Generator):
	"""Generates a loop fusion transformation"""
	def generate(self, individual):
		genes = []
		for m in individual.ws.fun:
			if called(m):
				for l in m.loops()[:-1]:
					genes.append(Gene(Transfo(m.name, "FORCE_LOOP_FUSION", loop=l.label)))
		return genes

class UnfoldGenerator(Generator):
	def generate(self, individual):
		""" Otherwise let's lookup everything. """
		genes = []
		for module in individual.ws.fun:
			if module.callees() and called(module):
				genes.append(Gene(Transfo(module.name, "UNFOLDING")))
		return genes

class InlineGenerator(Generator):
	def generate(self, individual):
		""" Otherwise let's lookup everything. """
		genes = []
		for module in individual.ws.fun:
			if called(module):
				for caller in module.callers():
					genes.append(Gene(Transfo(module.name, "INLINING", inlining_purge_labels=True,
						inlining_callers=caller.name)))
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
		self.living=True

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
			if self.args.test: runner = [self.args.test,exec_out]
			else: runner = [exec_out]

			elapsed=[]

			#print "running",exec_out
			for i in range(0,self.args.get_time):
				t=-1
				while t<0:
					if subprocess.Popen(runner, stdout=file(os.devnull),stderr=subprocess.STDOUT).wait() == 0:
						with open(workspace_gettime.STAMPFILE, "r") as f:
							t = f.readline()
						os.remove(workspace_gettime.STAMPFILE)
					else:
						raise Exception("Failed to run test, check your test file")
				elapsed+=[int(t)]
			elapsed.sort()
			self.execution_time=elapsed[len(elapsed)/2]
			self.min_time = elapsed[0]
			self.max_time = elapsed[-1]
			if not self.args.blork:os.remove(exec_out)

	def clone(self):
		individual=Individual(self.args)
		for igene in self.genes:
			individual.push(igene)
		return individual

	def rip(self):
		if self.living:
			self.ws.close()
			self.living=False
	
	def __str__(self):
		s=reduce(lambda x,y:x+" " +y,self.args.sources,"sources:")
		s+= " out:"+self.foutname+"\n"
		s+= "workspace:"+self.ws.name+"\n"
		s+= "execution time:" + str(self.execution_time) + " (min: {0}, max: {1})\n".format(self.min_time, self.max_time)
		for t in self.genes:
			s+=str(t)
		return s

	def __hash__(self):
		return reduce(xor,[ x.__hash__() for x in self.genes ],42)

	def __cmp__(self,other):
		return cmp(self.__hash__(),other.__hash__())
	
#
##
#

class Algo(object):
	def __init__(self,args):
		self.nbgen=args.gens
		self.args=args
		self.step=0
		eve=Individual(args)# we start with a dummy individual
		eve.rate()
		self.pool=set([eve])# who said eve was dummy ? because she eat the apple, may be ? 
	
	def run(self):
		# process nbgen generation
		#pdb.set_trace()
		while self.step < self.nbgen:
			self.msg()
			self.populate()
			self.rate()
			self.select()
			self.step+=1
		# make sure all remote workspace are closed
		for individual in self.pool:individual.rip()
		return self.sort()

	def rate(self):
		for individual in self.pool:
			individual.rate()

	def msg(self):
		print "Step %d: best element score is '%d'" % (
				self.step, min(self.pool,key=operator.attrgetter('execution_time')).execution_time )

	def sort(self):
		return sorted(self.pool,key=operator.attrgetter('execution_time'))

class Full(Algo):
	def __init__(self,args):
		super(Full,self).__init__(args)

	def populate(self):
		newpool=set()
		for individual in self.pool:
			if len(individual.genes) == self.step:
				geneCandidates=[]
				for generator in self.args.generators:
					geneCandidates+=generator.generate(individual)
				for gene in geneCandidates:
					newindividual=individual.clone()
					newindividual.push(gene)
					if newindividual in newpool or newindividual in self.pool:
						newindividual.rip()
					else: newpool.add(newindividual)
				individual.rip()
		self.pool|=newpool

	def select(self):
		pass



class Greedy(Algo):
	def __init__(self,args):
		super(Greedy,self).__init__(args)

	def populate(self):
		newpool=set()
		for individual in self.pool:
			geneCandidates=[]
			for generator in self.args.generators:
				geneCandidates+=generator.generate(individual)
			for gene in geneCandidates:
				newindividual=individual.clone()
				newindividual.push(gene)
				if newindividual in newpool or newindividual in self.pool:
					newindividual.rip()
				else: newpool.add(newindividual)
		self.pool|=newpool

	def select(self):
		eugenism=self.sort()
		for individual in eugenism[self.args.popsize:]:
			individual.rip()
		self.pool=set(eugenism[0:self.args.popsize])

#
##
#

class Genetic(Algo):
	def __init__(self,args):
		super(Genetic,self).__init__(args)
		# init population
		adam=Individual(self.args)
		geneCandidates=[]
		for generator in self.args.generators:
			geneCandidates+=generator.generate(adam)
		random.shuffle(geneCandidates)
		for gene in geneCandidates:
			individual=adam.clone()
			individual.push(gene)
			if individual in self.pool: individual.rip()
			else: self.pool.add(individual)
			if len(self.pool) == self.args.popsize:
				break
		self.rate()

	def populate(self):
		newpool=set()
		for individual in self.pool:
			geneCandidates=[]
			for generator in self.args.generators:
				geneCandidates+=generator.generate(individual)
			# add a random gene among the existing one
			random.shuffle(geneCandidates) 
			for gene in geneCandidates:
				newindividual=individual.clone()
				newindividual.push(gene)
				if newindividual in self.pool or newindividual in newpool:
					newindividual.rip()
				else:
					newpool.add(newindividual)
					break
		self.pool.update(newpool)

	def select(self):
		half_individual=set(random.sample(self.pool,len(self.pool)/2))
		other_half_individual=self.pool.difference(half_individual)
		def select_individual(i0,i1):
			# handle None case
			if not i0: return i1
			if not i1: return i0
			# other cases
			if i0.execution_time < i1.execution_time:
				if self.args.verbose: print i0,"<<<<<<<<<<",i1
				i1.rip()
				return i0
			else:
				if self.args.verbose: print i0,">>>>>>>>>>",i1
				i0.rip()
				return i1
		# both population may not have exactly the same number of elements,
		# this is taken into account in select_individual
		self.pool=set(map(select_individual,half_individual,other_half_individual))
	
#
##
#

class UnitTest:
	""" this one is in charge of performing some regression testing"""
	def __init__(self,args):
		self.args=args

	def check(self):
		#self.check_sequence_and_quit()
		self.check_newborn_hash()
		self.check_transfo_hash()
		self.check_gene_hash()
		self.check_grownup_hash()
		self.check_grownup_hash2()
		self.check_set_behavior()

	def check_newborn_hash(self):
		"""check if two new born have similar hashes"""
		adam=Individual(self.args)
		eve=Individual(self.args)
		try:
			if adam.__hash__() != eve.__hash__():
				raise Exception("Individual with no genes have different hash")
		finally:
			adam.rip()
			eve.rip()

	def check_transfo_hash(self):
		""" check if two transfo with same arg have same hash """
		t0=Transfo("main", "UNFOLDING",unfolding_callers="foo")
		t1=Transfo("main", "UNFOLDING",unfolding_callers="foo")
		if t0.__hash__() != t1.__hash__():
			raise Exception("Transformations with same parameters have different hash")
	
	def check_gene_hash(self):
		""" check if two gene with same arg have same hash """
		g0=Gene(Transfo("main", "UNFOLDING"))
		g1=Gene(Transfo("main", "UNFOLDING"))
		if g0.__hash__() != g1.__hash__():
			raise Exception("Genes with same parameters have different hash")

	def check_grownup_hash(self):
		""" check if two individuals with one same gene have similar hashes"""
		adam=Individual(self.args)
		eve=Individual(self.args)
		adam.push(Gene(Transfo("main", "UNFOLDING")))
		eve.push(Gene(Transfo("main", "UNFOLDING")))
		try:
			if adam.__hash__() != eve.__hash__():
				raise Exception("Individual with same genes have different hash")
		finally:
			adam.rip()
			eve.rip()
	
	def check_grownup_hash2(self):
		""" check if two individuals with the very same gene have similar hashes"""
		adam=Individual(self.args)
		eve=Individual(self.args)
		g0=Gene(Transfo("main", "UNFOLDING"))
		adam.push(g0)
		eve.push(g0)
		try:
			if adam.__hash__() != eve.__hash__():
				raise Exception("Individual with same genes have different hash")
		finally:
			adam.rip()
			eve.rip()

	def check_set_behavior(self):
		""" check if two individuals with one same gene can be inserted in the same set"""
		adam=Individual(self.args)
		eve=Individual(self.args)
		adam.push(Gene(Transfo("main", "UNFOLDING")))
		eve.push(Gene(Transfo("main", "UNFOLDING")))
		paradise=set()
		paradise.add(adam)
		try:
			if adam.__hash__() != eve.__hash__():
				raise Exception("Individual with same genes have different hash")
			if eve not in paradise:
				raise Exception("Individual with same hash can go in the same set")
		finally:
			adam.rip()
			eve.rip()

	def check_sequence_and_quit(self):
		""" check a particular transformation sequence """
		adam=Individual(self.args)
		adam.push(Gene(Transfo("main","UNFOLDING")))
		eve=adam.clone()
		for gene in adam.genes:eve.push(gene)
		eve.push(Gene(Transfo("main","FLATTEN_CODE")))
		baby=eve.clone()
		for gene in eve.genes:baby.push(gene)
		genes=FusionGenerator().generate(baby)
		for g in genes: print str(g)
		baby.push(genes[1])
		baby.ws.fun.main.display()
		sys.exit(2)

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
			}
	
	for x in parser.items("generators"):
		if x[0] not in base:
			print "Error when parsing config file, as " + x[0] + " is not a generator"
		else:
			args.generators += [base[x[0]]]
	
	def parseGenerator(args,generator,x):
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
		args.generators.append(generator(x[0], **propvals))
	
	try:
		for x in parser.items("dummyGenerators"):
			parseGenerator(args,DummyGenerator,x)
	except ConfigParser.NoSectionError:pass
	try:
		for x in parser.items("uniqGenerators"):
			parseGenerator(args,UniqGenerator,x)
	except ConfigParser.NoSectionError:pass
	
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
	#parser.add_option('--crossovers', default=1, type=int, help='Number of crossovers to perform each generation')
	#parser.add_option('--tournaments', default=3, type=int, help='Number of winners of a tournament for the genetic algorithm')
	parser.add_option('--popsize', type=int, default=4,help='Number of individuals for the genetic algorithm')
	parser.add_option('--cppflags', default='', help='Optional added arguments to the compiler')
	parser.add_option('--test',help='Optionnal arguments for benchmarking code',dest='test')
	parser.add_option('--bench-iter', default=50,type=int, help='Number of iteration for benchmarking',dest="get_time")
	parser.add_option('--unit-test', action="store_true", help='perform some unit test',dest="unit_test")
	parser.add_option('--out', help='directory to store result in')
	parser.add_option('--blork', action="store_true", help='Leave a real mess of temporary files, usefull for debug')
	parser.add_option('-v', action="store_true", help='be very talkative',dest="verbose")
	
	(args, files) = parser.parse_args()
	args.sources = files

	if args.unit_test:UnitTest(args).check()

	# verify some settings
	if not args.sources:
		raise Exception("You need to input at least one source file")
	if not type(args.sources) is list: args.sources=[args.sources]
	available_algos = { "genetic": Genetic , "greedy": Greedy, "full": Full}
	args.algo=available_algos[args.algo]

	if args.test:
		if not os.path.isfile(args.test):
			raise Exception("Test file '%s' does not exist" % args.test)
		elif not os.access(args.test,os.X_OK):
			raise Exception("Test file '%s' not executable" % args.test)
	else:
		print "No test file provided, defaulting to bare a.out"

	if args.out:
		outdir=args.out
		if os.path.exists(outdir):
			print "Warning, {0} already exist".format(outdir),
		counter=0
		while os.path.exists(outdir):
			outdir="{0}{1}".format(args.out,counter)
			counter+=1
		args.out=outdir
		os.mkdir(args.out)
		print ", using {0} instead".format(outdir)
	else:
		args.out=tempfile.mkdtemp(dir=".",prefix="pypsearch")
		print "No output dir provided, defaulting to {0}".format(args.out)
		
	if args.log:
		args.log = open(args.log, 'w')

	os.environ["PIPS_CPP_FLAGS"]=args.cppflags

	ParseConfigFile(args)
	return args

def pypsearch():
	print "Initializing Pypsearch"
	args=ParseCommandLine()
	
	# init 
	workspacedir = getwdir(args.sources)
	if os.path.exists(workspacedir):
		shutil.rmtree(workspacedir)
	os.mkdir(workspacedir)
	[ shutil.copy(x,workspacedir) for x in args.sources	]
	
	#
	# launch algo
	results=[]
	print "Running %s algorithm" %  args.algo.__name__
	results=args.algo(args).run()
	print "Best element in {0} with score {1}".format(args.out,results[0].execution_time)

	
	bestsources=[ os.path.join(workspacedir,results[0].foutname, os.path.basename(x)) for x in args.sources ]
	[ shutil.copy(x,args.out) for x in bestsources ]
	
	if args.log:
		print >> args.log , "-- best results" 
		original =  Individual(args)
		original.rate()
		original.rip()
		for r in results:
			print >> args.log, r
			print >> args.log, "--"
		print >> args.log, "-- original result --"
		print >> args.log, original
		print >> args.log, "done !"
	
	if not args.blork:
		shutil.rmtree(workspacedir)

	

if __name__ == "__main__":
	pypsearch()
