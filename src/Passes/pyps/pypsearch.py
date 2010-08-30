#!/usr/bin/env python
from copy import deepcopy
import pyps
import workspace_gettime
import lazy_workspace
from lazy_workspace import transfo
from lazy_workspace import property
import sac
#import random_debug as random
import random
from optparse import OptionParser
import operator
import re # we are gonne use regular expression
import string
import time
import sys
from sys import argv
from exceptions import Exception
import subprocess
import os
import shutil
import socket
import cPickle
import pyrops
import ConfigParser

"""
This module provide three ways of exploring the transformation space for a given module in a given programm
- brute force exploration
- greedy exploration
- genetic algorithm based exploration
"""

#
##
#

class genome:
	def __init__(self, genes):
		self.genes = genes
	
	def __hash__(self):
		return hash(tuple(self.genes))
	
	def __cmp__(self, other):
		return cmp(self.genes, other.genes)

class gene:
	"""a gene contains a transformation with associated properties
	it represents a parametrized transformation on a module"""
	def __init__(self,*codons):
		self.codons=codons
		
	""" forced set to true prevents lazy behaviour and force the workspace to
		execute the code right now. """
	def run(self, wsp, forced=False):
		"""apply all the transformation in the gene on module `name'"""
		map(lambda x:wsp.run(x),self.codons)
		if forced:
			wsp.detach()

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
	
	def __hash__(self):
		return hash(tuple(self.codons))
	
"""Generates a random gene"""
def generateRandomGene(workspace, past_genes = None):
	return random_items(generateAllGenes(workspace, past_genes), 1)[0]

"""Generates all genes"""	
def generateAllGenes(workspace, past_genes = None):
	return reduce(lambda x,y: x + y.generate(workspace, past_genes), getGenerators(), [])
	#return [gene("linpack!daxpy_r")];
	
def getStartingSACGenome(module):
	"""transfoStrings = ["SPLIT_UPDATE_OPERATOR", 
		"IF_CONVERSION_INIT", "IF_CONVERSION", "IF_CONVERSION_COMPACT",
		"PARTIAL_EVAL", "SIMD_ATOMIZER", "SIMDIZER_AUTO_UNROLL", "PARTIAL_EVAL",
		"CLEAN_DECLARATIONS", "SUPPRESS_DEAD_CODE", "SIMD_REMOVE_REDUCTIONS", 
		"SINGLE_ASSIGNMENT", "SIMDIZER", "SIMD_LOOP_CONST_ELIM", "CLEAN_DECLARATIONS",
		"SUPPRESS_DEAD_CODE"]
		
	genes = [gene(transfo(module, x)) for x in transfoStrings]
	
	return genes"""
	return []

class generator(object):
	def __init__(self):
		self.generated = {}
	
	def fastGenerating(self, g):
		if g in self.generated:
			return self.generated[g]
		return None

class unroll_gen(generator):
	"""Generates a partialeval transformation"""
	def generate(self, worksp, past_genes):
		""" past_genes of None means no assumptions """
		if past_genes is not None:
			g = genome(past_genes)
			tentative = self.fastGenerating(g)
			if tentative:
				return tentative
		
		loops = worksp.getAllLoops()
		
		genes = []
		for loop in loops:
			rate = 1
			
			for i in range(3):
				rate = rate * 2
				genes = genes + [gene(transfo(loop.module.name, "UNROLL", loop=loop.label, unroll_rate=rate))]
		return genes

class red_var_exp_gen(generator):
	"""Generates a partialeval transformation"""
	def generate(self, worksp, past_genes):
		""" past_genes of None means no assumptions """
		if past_genes is not None:
			g = genome(past_genes)
			tentative = self.fastGenerating(g)
			if tentative:
				return tentative
		
		loops = worksp.getAllLoops()
		
		genes = []
		for loop in loops:
			genes = genes + [gene(transfo(loop.module.name, "REDUCTION_VARIABLE_EXPANSION", loop = loop.label))]
		return genes

class interchange_gen(generator):
	"""Generates a loop interchange transformation"""
	def generate(self, worksp, past_genes):
		""" past_genes of None means no assumptions """
		if past_genes is not None:
			g = genome(past_genes)
			tentative = self.fastGenerating(g)
			if tentative:
				return tentative
		
		genes = []
		
		loops = worksp.getAllLoops()
		
		for loop in loops:
			if len(loop.loops()) > 0:
				genes = genes + [gene(transfo(loop.module.name, "LOOP_INTERCHANGE", loop=loop.label))]
		return genes

class not2inarow_gen(generator):
	def __init__(self, name, **args) :
		self.name = name
		self.properties = args
		super(not2inarow_gen, self).__init__()
	
	"""Generates a partialeval transformation"""
	def generate(self, worksp, past_genes):
		""" past_genes of None means no assumptions """
		if past_genes is not None:
			g = genome(past_genes)
			tentative = self.fastGenerating(g)
			if tentative:
				return tentative
		
		if len(past_genes) == 0:
			last_gene = None
		else:
			last_gene = past_genes[-1]

		genes = []
		for module in worksp:
			if module.name[0:4] == "SIMD":
				continue
			g = gene (transfo(module.name, self.name, **self.properties))
			if g is not last_gene:
				genes = genes + [g]
			
		return genes


class inline_gen(generator):
	def generate(self, worksp, past_genes):
		
		""" past_genes of None means no assumptions """
		if past_genes is not None:
			g = genome(past_genes)
			tentative = self.fastGenerating(g)
			if tentative:
				return tentative
				
			""" If the [] already exists, it means the base genes exist, and then
			it's assumed that no other genes for inlining were introduced since.
			
			Plus it would be useless to have two same genes in a chromosome, and so
			it allows us to do a fast lookup without even looking at the workspace"""
			base = self.fastGenerating(genome([]))
			if (base):
				ret = deepcopy(base)
				for x in past_genes:
					if x in ret:
						ret.remove(x)
				""" Saving result for future use """
				self.generated[g] = ret
				return ret

		""" Otherwise let's lookup everything. """
		genes = []
		for module in worksp:
			if module.name[0:4] == "SIMD":
				continue
			for caller in module.callers():
				genes = genes + [gene(transfo(module.name, "INLINING", inlining_purge_labels=True,
					inlining_callers=caller))]
		
		if past_genes is not None:
			""" Saving result for future use """
			self.generated[genome([])] = genes

		return genes

"""class stripmine_gen:
	#Generates all the transformations related to loop stripmining
	@staticmethod
	def generate(worksp):
		genes = []
		
		for loop in getAllLoops(worksp.filter()):
			rate = 1
			for i in range(4):
				rate = rate * 2
				genes = genes + [gene(loop.module.name, property("LOOP_LABEL", loop.label), 
									  property("STRIP_MINE_KIND", 0),
									  property("STRIP_MINE_FACTOR", rate),
									  transfo("STRIP_MINE"))]
		return genes"""

"""class other_gens:
	@staticmethod
	def generate(worksp):
		genes = []
		for loop in getAllLoops(worksp.filter()):
			rate = 64
			for i in range(4):
				genes = genes + [gene(loop.module.name, property("LOOP_LABEL", loop.label),
					property("INDEX_SET_SPLITTING_BOUND",rate), transfo("INDEX_SET_SPLITTING"))]
				rate = rate * 2
				genes = genes + [gene(loop.module.name, property("LOOP_LABEL",loop.label), 
					transfo("LOOP_INTERCHANGE"))]
		for module in worksp:
			genes = genes + [gene(module.name, transfo("LOOP_NORMALIZE")), gene(module.name, 
				transfo("SUPPRESS_DEAD_CODE")), gene(module.name, transfo("COARSE_GRAIN_PARALLELIZATION")), 
				gene(module.name, transfo("ARRAY_TO_POINTER")), gene(module.name, transfo("SIMD_LOOP_CONST_ELIM")),
				gene(module.name, transfo("INVARIANT_CODE_MOTION")), gene(module.name, transfo("SCALARIZATION"))]
		return genes"""

inline = inline_gen()
unroll = unroll_gen()
red_var_exp = red_var_exp_gen()
interchange = interchange_gen()

generators = []
try:
	parser = ConfigParser.RawConfigParser()
	parser.readfp(open("pypsearch.cfg"))

	generatorsD = parser.items("generators")
	basicGeneratorsD = parser.items("basicGenerators")

	corr = {"inline":inline, "unroll":unroll, "red_var_exp":red_var_exp, "interchange":interchange}

	for x in generatorsD:
		if x[0] not in corr:
			print "Error when parsing config file, as " + x[0] + " is not a generator"
		else:
			generators += [corr[x[0]]]

	for x in basicGeneratorsD:
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
		
		generators += [not2inarow_gen(x[0], **propvals)]
except:
	pass

if generators is []:
	print "No generator given in the config file (pypsearch.cfg), so using only the inline generator"
	generators = [inline]

def getGenerators():
	#return [inline, unroll, partialeval, interchange, ompify]
	#return [unroll, partialeval, interchange, ompify, privatize, substitute]
	"""sac_gens = [split_update, if_conv_init, if_conv, if_conv_compact, simd_atomizer, simdizer_auto_unroll, clean_decl,
		suppress_dead_code, simd_remove_reductions, single_assignment, simdizer, simd_loop_const_elim]"""
	"""return [unroll, red_var_exp, substitute, partialeval, simd_atomizer, single_assignment,
		simdizer, simd_loop_const_elim, inline]"""
	#return [inline]
	return generators
#
##
#

def random_items(arr,n):
	""" get `n' random items from an array """
	l = len(arr) -1
	iout,out=[],[]
	if n == l: return arr
	elif n < l:
		for i in range(0,n):
			t=random.randint(0,l)
			while t in iout:t=random.randint(0,l)
			iout.append(t)
		for i in iout:out.append(arr[i])
	else:
		for i in range(0,n):
			out.append(arr[random.randint(0,l)])

	return out
		
		
class chromosome(object):
	valid_chromosomes = {}
		
	def __init__(self, sources, get_time=20, genes = [], original=False, sse=False):
		self.genome = genes
		self.sources = sources
		self.original = original
		self.workspace = None
		self.get_time=get_time
		self.execution_time=0.0
		self.sse = sse
		self.foutname=string.join([str(random.randint(0,1000000)) , os.path.basename(self.sources[0])], "_")
	
	def createWorkspace(self):
		if self.workspace:
			raise RuntimeError("Workspace already existing")

		if self.sse:
			w= pyrops.pworkspace(self.sources, parents = [sac.workspace, lazy_workspace.workspace, workspace_gettime.workspace])
		else:
			w= pyrops.pworkspace(self.sources, recoverInclude = False, parents = [lazy_workspace.workspace, workspace_gettime.workspace])

		w.activate("MUST_REGIONS")
		w.activate("PRECONDITIONS_INTER_FULL")
		w.activate("TRANSFORMERS_INTER_FULL")
		w.activate("RICE_SEMANTICS_DEPENDENCE_GRAPH")
		w.activate("RICE_REGIONS_DEPENDENCE_GRAPH")
		w.activate("REGION_CHAINS")

		w.set_property(RICEDG_STATISTICS_ALL_ARRAYS=True)
		w.set_property(FOR_TO_DO_LOOP_IN_CONTROLIZER=True)
		w.set_property(C89_CODE_GENERATION=True)
		w.set_property(CONSTANT_PATH_EFFECTS=False)
		w.set_property(INLINING_USE_INITIALIZATION_LIST=False)
		w.set_property(SIMDIZER_AUTO_UNROLL_SIMPLE_CALCULATION = False)
		w.set_property(SIMDIZER_AUTO_UNROLL_MINIMIZE_UNROLL = False)
		w.set_property(PRETTYPRINT_ALL_DECLARATIONS = True)
		w.set_property(SIMD_FORTRAN_MEM_ORGANISATION = False)
		w.set_property(SAC_SIMD_REGISTER_WIDTH = 128)
		self.workspace = w
		
	def closeWorkspace(self):
		self.workspace.close()
		self.workspace = None
		
	def applyNGenes(self, begin=0, ngenes=-1):
		if ngenes == -1:
			ngenes = self.genomeLength()

		for i in range(begin, ngenes):
			#generates all the labels for the loops in the workspace, so that changes
			#with those labels can be applied
			#getAllLoops(self.workspace.filter())
			self.genome[i].run(self.workspace)
	
	def computeTime(self):
		self.createWorkspace()
		self.applyNGenes()
		
		w = self.workspace
				
		wdir=getwdir(self.sources) +os.sep+self.foutname
		w.save(indir=wdir)
		
		cflags=os.environ["PIPS_CPP_FLAGS"]+" -I. -O3 -fopenmp"
		if not self.original:
			cflags = cflags + " -fno-inline"
		#random names in case several processes are running at the same time
		randString = str(random.randint(0, 1000000)) + str(random.randint(0, 1000000))
		if not self.sse:
			exec_out = w.compile(CC="gcc", CFLAGS=cflags,outdir=wdir,outfile="/tmp/" + randString + ".out")
		else:
			exec_out = w.sse_compile(CC="gcc", CFLAGS=cflags,outdir=wdir,outfile="/tmp/" + randString + ".out")
		elapsed=[]
		#print "running",exec_out
		for i in range(0,self.get_time):
			t=-1
			while t<0:
				subprocess.call(exec_out)
				with open("_pyps_time.tmp", "r") as f:
					t = f.readline()
				#print "measured:" , t
			elapsed+=[int(t)]
		elapsed.sort()
		self.execution_time=elapsed[len(elapsed)/2]
		self.min_time = elapsed[0]
		self.max_time = elapsed[-1]
				
		self.closeWorkspace()
	
	def genomeLength(self):
		return len(self.genome)
	
	#Runs an automated procedure to replace genes no longer valid by new ones
	# Doesn't need a workspace if it is fully corrected (from first gene)
	# it assumes previous genes have already been run
	def autoCorrect(self, begin=0):
		""" No need to check again if already checked """
		if self in type(self).valid_chromosomes:
			return
			
		wspGenerated = False
		if (self.workspace == None):
			self.createWorkspace()
			wspGenerated = True
		
		for i in range(begin, self.genomeLength()):
			#generates all the labels for the loops in the workspace, so that changes
			#with those labels can be applied
			#getAllLoops(self.workspace.filter())
			
			try:
				self.genome[i].run(self.workspace, forced=True)
			except RuntimeError:
				#even if a transformation is available, it may fail, like 
				self.genome[i] = gene()
		
		#Removes useless genes
		self.genome = [x for x in self.genome if x is not gene()]
		
		if (wspGenerated):
			self.closeWorkspace()
		
		type(self).valid_chromosomes[self] = True
	
	#Adds a random gene in a random place of the chromosome, and fix the remaining
	#if invalidated
	def mutate(self):
		mutationIndex = random.randint(0, self.genomeLength())
		
		self.createWorkspace()
		self.applyNGenes(0, mutationIndex)
		self.genome.insert(mutationIndex, generateRandomGene(self.workspace, self.genome[0:mutationIndex]))
		
		if self in type(self).valid_chromosomes:
			self.closeWorkspace()
			return

		self.autoCorrect(mutationIndex)
		self.closeWorkspace()
		
	@staticmethod
	def crossover(chr1, chr2):
		if chr1.genomeLength() == 0 or chr2.genomeLength() == 0:
			return
		index1 = random.randint(0, chr1.genomeLength()-1)
		index2 = random.randint(0, chr2.genomeLength()-1)

		#swaps the genes
		(chr1.genome[index1], chr2.genome[index2]) = (chr2.genome[index2], chr1.genome[index1] )
		#fix the invalid things that may have happened
		chr1.autoCorrect()
		chr2.autoCorrect()
			
	def __getitem__(self,i):
		return self.genome[i]

	def __setitem__(self,i):
		return self.genome[i]
		
	def __str__(self):
		s=reduce(lambda x,y:x+" " +y,self.sources,"sources:")
		s+= " out:"+self.foutname+"\n"
		s+= "execution time:" + str(self.execution_time) + " (min: {0}, max: {1})\n".format(self.min_time, self.max_time)
		for t in self.genome:
			s+=str(t)
		return s


class genetic:
	"""use genetic algorithm to explore the transformation space"""
	def __init__(self,sources,args):
		if args.module is not '':
			self.runOnSac = True
			self.sacModule = args.module
			w = pyrops.pworkspace(sources, parents=[sac.workspace, lazy_workspace.workspace])
		else:
			self.runOnSac = False
			w = pyrops.pworkspace(sources, parents=[lazy_workspace.workspace])

		self.base_genes= generateAllGenes(w, [])
		w.close()
		
		self.sources=sources
		self.nbgen=args.gens
		if (args.popsize):
			self.popsize = args.popsize
		else:
			self.popsize=len(self.base_genes)
			
		nbtournament = args.tournaments
		nbcrossovers = args.crossovers
		
		self.nbtournament=min(nbtournament,self.popsize/3)
		self.nbcrossovers=min(nbcrossovers,self.popsize/3)
	
		if args.server:
			print "Starting to listen to port " + str(args.port) + "..."
			self.listener = socket.socket()
			self.listener.bind(('localhost', args.port))
			self.listener.listen(5)
			print "Listening!"
			
			count = args.slaves
			
			self.totalSocks = count
			self.socks = []
			
			while count > 0:
				print "Waiting for a client to connect... Still " + str(count) + " left."
				self.socks.append(self.listener.accept()[0])
				count = count - 1
				print "New connection accepted!"
			print "All Clients Connected!"
			
			cmd = "sources"
			sourcesToSend = []
			for x in args.sources:
				with open(x, 'r') as f:
					content = f.read()
				sourcesToSend += [(x,content)]
			
			msg = cmd + "||" + cPickle.dumps(sourcesToSend)
			
			for x in range(0, self.totalSocks):
				print "Sending sources to socket " + str(x)
				sendNetworkMessage(self.socks[x], msg)
			print "Source files sent to all sockets!"
				
			cmd = "flags"
			val = os.environ["PIPS_CPP_FLAGS"]
			msg = cmd + "||" + cPickle.dumps(val)
			
			for x in range(0, self.totalSocks):
				print "Sending flags to socket " + str(x)
				sendNetworkMessage(self.socks[x], msg)
			print "Flags sent to everyone"
			
			cmd = "headers"
			filesToSend = []
			# Look for additional header files. For now, header files are only the ones
			# in the folders specified by '-Ixxx'
			folders = re.findall(r'(?<=-I)\S+', os.environ["PIPS_CPP_FLAGS"])
			
			if not (folders is []):
				for folder in folders:
					for root, dirs, files in os.walk(folder):
						for file in files:
							#only send header files
							if file[-2:] == ".h":
								path = os.path.join(root, file)
								with open(path, 'r') as f:
									content = f.read()
								filesToSend += [(path, content)]
				if not (filesToSend == []):
					msg = cmd + "||" + cPickle.dumps(filesToSend)

					for x in range(0, self.totalSocks):
						print "Sending additional headers to socket " + str(x)
						sendNetworkMessage(self.socks[x], msg)
					print "Additional headers sent to all sockets!"
					
			
			cmd = "options"
			if self.runOnSac:
				vals = ["sse"]
			else:
				vals = []
			msg = cmd + "||" + cPickle.dumps(vals)

			for x in range(0, self.totalSocks):
				print "Sending options to socket " + str(x)
				sendNetworkMessage(self.socks[x], msg)
			print "Options sent to everyone"
		else:
			self.listener = None
		self.server = args.server
		
	def run(self):
		"""do the job over nbgen steps"""
		# first generation
		self.file = open('profiling', 'w+')
		new_generation = self.birth(self.popsize)

		if not self.server:
			self.computeFitness(new_generation)
			new_generation.sort(key=operator.attrgetter('execution_time'))

		# evaluate each generation
		for gen in range(0,self.nbgen):
			#Make an array with the same references as the older one, but separated
			old_generation = [x for x in new_generation]
			new_generation = [deepcopy(old_generation[0])]
			# Number of new birth, random mutation, random crossover
			numberOfSideBirths = self.nbcrossovers + 1 + 1
			winners= [deepcopy(x) for x in self.tournament(old_generation)]
			new_generation += winners
			# cross overs
			crossovers=self.crossover(winners)
			new_generation += crossovers
			# mutation
			mutants=self.mutate(winners)
			new_generation += mutants
			# one mutation and crossover for the losers
			births = self.birth(1) + random_items(old_generation, max(1, self.popsize - len(new_generation)))
			
			index = random.randint(0, len(births) -1)
			births[index] = self.mutate([births[index]])[0]
			#not necessary test
			if len(births) >= 2:
				indexes = random_items(range(0, len(births)), 2)
				(births[indexes[0]], births[indexes[1]]) = 				\
					(deepcopy(births[indexes[0]]), deepcopy(births[indexes[1]]))
				chromosome.crossover(births[indexes[0]], births[indexes[1]])
			new_generation += births
			
			# fitness
			if not self.server:
				self.computeFitness(new_generation)
				new_generation.sort(key=operator.attrgetter('execution_time'))
		# end
		return new_generation

	def computeFitness(self,generation):
		print ("computing fitness ...")
		print >> self.file, "Computing fitness /" + str(time.time())
		for x in generation: x.computeTime()
		print >> self.file, "Computing fitness end /" + str(time.time())

	def tournament(self,generation,nb=0):
		"""perform `select' battles between element from `fitness' and 
		return a list of winners"""
		if nb == 0: nb=self.nbtournament
		winners=[]
		
		if not self.server:
			#The items in generation are already sorted by execution time
			return generation[0:nb]
		else:
			#Don't put more sockets than there is winners
			totalSocks = min(self.totalSocks, nb)
			
			winnersPerRemoteComputer = [nb / totalSocks for x in range(0, totalSocks)]
			chromosomesPerRemoteComputer = [len(generation) / totalSocks for x in range(0, totalSocks)]
			remains = nb - (nb / totalSocks) * totalSocks
			chRemains = len(generation) - (len(generation) / totalSocks) * totalSocks
			
			#As the division might not be perfect, there may be leftovers item that will be given to the first few
			#chromosomes
			index = 0
			
			while remains > 0:
				remains = remains - 1
				winnersPerRemoteComputer[index] = winnersPerRemoteComputer[index]  + 1
			
			index = 0
			while chRemains > 0:
				chRemains = chRemains - 1
				chromosomesPerRemoteComputer[index] = chromosomesPerRemoteComputer[index]  + 1
			
			#To avoid the phenomenon of winners gathered in the first few genes, as the winners from the tournament
			#are always put in the first genes
			random.shuffle(generation)
			
			absindex = 0
			#now send the message to the clients, asking for the winners
			for x in range(0, totalSocks):
				if chromosomesPerRemoteComputer[x] == 0:
					continue

				cmd = "tournament"
				winnersNum = winnersPerRemoteComputer[x]
				chr = generation[absindex:absindex + chromosomesPerRemoteComputer[x]]
				absindex += chromosomesPerRemoteComputer[x]
				
				cmdstring = cmd + "||" + cPickle.dumps((winnersNum, chr))
				sendNetworkMessage(self.socks[x], cmdstring)
			#now receiving the results
			for x in range(0, totalSocks):
				if winnersPerRemoteComputer[x] == 0:
					continue
				answer = recvNetworkMessage(self.socks[x])
				winners += cPickle.loads(answer)

		return winners

	def crossover(self,winners):
		"""perform crossovers betwwen winners of tournaments"""
		crossovers=[]
		print ("cross overs ...")
		for i in range(0,self.nbcrossovers):
			xmens=random_items(winners,2)
			#Deep copy so that regular genes are not affected by the changes
			xmens[0], xmens[1] = deepcopy(xmens[0]), deepcopy(xmens[1])
			chromosome.crossover(xmens[0], xmens[1])
		print ("adding", str(len(crossovers)), "crossovers")
		return crossovers

	def mutate(self,winners):
		"""perform random mutation on winners"""
		print ("mutation ...")
		mutants= [deepcopy(x) for x in winners]
		map(lambda x: x.mutate(), mutants)
		print ("adding", str(len(mutants)), "mutation")
		return mutants

	def birth(self,nb):
		"""create `nb' new elements from `base_genes'"""
		print ("births ...")
		print >> self.file, "Birthing " + str(nb) + " / " + str(time.time())
		if self.runOnSac:
			births = [chromosome(self.sources, genes = getStartingSACGenome(self.sacModule), sse=True) for x in range(0, nb)]
		else:
			births=map(lambda x: chromosome(self.sources, genes = [x]),random_items(self.base_genes,nb))
			""" Some transformations might not even be valid on first try, like loop interchange """
			for x in births:
				x.autoCorrect()
		print ("adding", str(len(births)), "births")
		print >> self.file, "End of birthing / " + str(time.time())
		return births
	
def getwdir(sources):
	return "WDIR_"+"".join("-".join(sources).split('/'))
	
def launchClient(host):
	print "launching client with host {0}.".format(host)
	
	sock = socket.socket()
	print "Attempting connecting..."

	(h, p) = host.split(":")
	p = int(p)

	sources = []
	sse = False
	
	while True:
		try:
			msg = recvNetworkMessage(sock).partition('||')
			cmd = msg[0]
			print "Received message with command '" + cmd + "'"
			obj = cPickle.loads(msg[2])
			
			if cmd == "sources":
				for x in obj:
					name = x[0]
					content = x[1]
					namepath = name.rpartition('/')
					
					#Making an already existing directory would raise an exception
					try:
						os.makedirs(namepath[0])
					except:
						pass
										
					with open(name, 'w') as f:
						f.write(content)
					
					print name + " added to sources"
					sources += [name]
			elif cmd == "headers":
				for x in obj:
					name = x[0]
					content = x[1]
					namepath = name.rpartition('/')
					
					#Making an already existing directory would raise an exception
					try:
						os.makedirs(namepath[0])
					except:
						pass
										
					with open(name, 'w') as f:
						f.write(content)
					
					print name + " added to headers"
			elif cmd == "options":
				if "sse" in obj:
					sse = True
				else:
					sse = False
			elif cmd == "tournament":
				winnersNum = obj[0]
				chrs = obj[1]

				for x in chrs:
					x.computeTime()

				chrs.sort(key=operator.attrgetter('execution_time'))
				
				winners = chrs[0:winnersNum]
				
				if len(winners) > 0:
					answer = cPickle.dumps(winners)
					sendNetworkMessage(sock, answer)
			elif cmd == "flags":
				os.environ["PIPS_CPP_FLAGS"]= obj
				print "Compilation flags set to " + obj
			else:
				print "No such command found"
			print "End Processing"
		except socket.error:
			while True:
				try:
					sock.connect((h, p))
					break
				except:
					print "Failed to connect, will try again in 10 seconds"
					time.sleep(10)
					pass
			print "Connected! Waiting orders."

def recvNetworkMessage(socket):
	length = 4
	buf = ''
	
	while len(buf) < length:
		buf = buf + socket.recv(length - len(buf))
	
	length = ord(buf[0]) * (256**3) + ord(buf[1]) * (256 ** 2) + ord(buf[2]) * 256 + ord(buf[3])
	
	buf = ''
	
	while len(buf) < length:
		buf = buf + socket.recv(length - len(buf))
		
	return buf
	
def sendNetworkMessage(socket, string):
	length = len(string)
	
	header = chr(length/(256**3)) + chr( (length%(256**3)) / (256**2)) + chr( (length%(256**2)) / 256) + chr(length%256)
	
	msg = header + string
	
	socket.send(msg)
	
#
##
#

#example of how to run it: 
# python xpc.py --algo=genetic --sources=linpack.c --CPPFLAGS="-lm" --restore=true
def main():
	parser = OptionParser(description="Pypsearch - Automated exploration of the set of transformations"
			+" with python.")
	parser.add_option('--CPPFLAGS', default='', help='Optional added arguments to the compiler')
	parser.add_option('--restore', default=0, type=int, help='Should we try to reproduce the same result as the session before?')
	parser.add_option('--log', help='log file to save the best results')
	parser.add_option('--gens', default=1, type=int, help='Number of generations for the genetic algorithm')
	parser.add_option('--crossovers', default=1, type=int, help='Number of crossovers to perform each generation')
	parser.add_option('--tournaments', default=3, type=int, help='Number of winners of a tournament for the genetic algorithm')
	parser.add_option('--popsize', type=int, help='Number of individuals for the genetic algorithm')
	parser.add_option('--module', default='', help='If specified, will run a SAC optimization on this module using Frederic Perrin\'s transformations as a base')
	parser.add_option('--host', default='', help='If specified, will try to connect to the host address given and obey the host to reduce its load. The host address should be IP:port, likeso: 127.0.0.1:1234')
	parser.add_option('--server', help='Acts as a server to use other computers to speed up the process.')
	parser.add_option('--port', default=1080, type=int, help='When acting as a server, listens to this port. --server must be specified too')
	parser.add_option('--slaves', type=int, help='Number of clients that will connect to this server.')
	
	(args, files) = parser.parse_args()
	
	if args.host is not '':
		launchClient(args.host)
		return 0

	args.sources = files
	
	if not args.sources:
		print ("Error, you need to input at least one source file")
		sys.exit(2)
	# init 
	if args.restore:
		random.restore("log_random")
		
	if args.log == None:
		args.log = sys.stdout
	else:
		args.log = open(args.log, 'w')

	#random.start_saving("log_random")
	
	os.environ["PIPS_CPP_FLAGS"]=args.CPPFLAGS
	os.environ["SEMANTICS_DEBUG_LEVEL"]="0"
	print ("PIPS_CPP_FLAGS=", os.environ["PIPS_CPP_FLAGS"], "--")
	print ("create xp directory")
	workspacedir = getwdir(args.sources)
	if os.path.exists(workspacedir):
		shutil.rmtree(workspacedir)
	os.mkdir(workspacedir)
	print ("copy input file to this dir")
	map(lambda x:shutil.copy(x,workspacedir),args.sources)
	
	#
	# launch algo
	results=[]
	# full transversal
	algo=genetic(args.sources,args)
	results=algo.run()
	

	for r in results:
		r.computeTime()
	
	# output
	results.sort(key=operator.attrgetter('execution_time'))

	print >> args.log , "-- best results" 
	
	if args.module:
		genes = getStartingSACGenome(args.module)
	else:
		genes = []

	if args.module:
		original =  chromosome(args.sources, genes = genes, sse=True, original=True)
	else:
		original =  chromosome(args.sources, genes = genes, original=True)
	original.computeTime()
	
	for r in results:
		print >> args.log, r
		print >> args.log, "--"
	print ("-- original result --")
	print original
	print ("done !")
	#random.finish_saving()

	

if __name__ == "__main__":
	main()
