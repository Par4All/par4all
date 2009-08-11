#!/usr/bin/env python
from copy import deepcopy
from pyps import *
import random
import getopt
import pypips
import operator
import re # we are gonne use regular expression
import random
import string
import time
import sys
from sys import argv
from exceptions import Exception
import os
import shutil

"""
This module provide three ways of exploring the transformation space for a given module in a given programm
- brute force exploration
- greedy exploration
- genetic algorithm based exploration
"""

#
##
#

import fileinput
class simd_workspace(workspace):
	"""extended pips workspace that make it eay to compile SAC generated code"""

	def __init__(self,*sources2):
		self.re1=re.compile("float v4sf_([^ ,]+)\[.*\]")
		self.re2=re.compile("v4sf_([^ ,]+)")
		workspace.initialize(self,list(sources2)+["simd.c"])

	def save(self,indir="",with_prefix=""):
		saved=workspace.save(self,indir,with_prefix)
		# filter simd.c
		outfiles=[]
		for l in saved:
			if os.path.basename(l) != "simd.c": outfiles+=[l]
		for o in outfiles:
			fi = fileinput.FileInput(o,inplace=1)
			first=False
			for line in fi:
				if not first:
					print '#include "sse.h"'
					first=True
				line=self.re1.sub(r"__m128 \1",line)
				line=self.re2.sub(r"\1",line)
				print line
			fi.close()
		return outfiles









#
##
#
class act:
	"""stores informations concerning a code activation"""
	def __init__(self,phase):
		self.phase=phase
	def __str__(self):return "activate:"+self.phase+"\n"
	def run(self,name):
		"""run this activation on module `name'"""
		pypips.activate(self.phase) 
	def __cmp__(self,other):
		if type(other).__name__ != 'act': return -1
		return cmp(self.phase,other.phase)

class transfo:
	"""stores informations concerning a code transformation"""
	def __init__(self,transfo):
		self.transfo=transfo
	def __str__(self):return "transformation:"+self.transfo+"\n"
	def run(self,name):
		"""run this transformation on module `name'"""
		pypips.apply(self.transfo,name) 
	def __cmp__(self,other):
		if type(other).__name__ != 'transfo': return -1
		return cmp(self.transfo,other.transfo)

class property:
	"""stores informations concerning a pips property"""
	def __init__(self,prop, val):
		self.prop=prop
		self.val=val
	def __str__(self):return "property:"+ self.prop+ " value:"+ self.val+"\n"
	def run(self,name):
		"""set the property on current workspace"""
		pypips.set_property(self.prop,self.val)
	def __cmp__(self,other):
		if type(other).__name__ != 'property': return -1
		n=cmp(self.prop,other.prop) 
		if n == 0: return cmp(self.val,other.val) 
		else: return n

class gene:
	"""a gene contains a transformation with associated properties
	it represents a parametrized transformation on a module"""
	def __init__(self,*codons):
		self.codons=codons
	def run(self,name):
		"""apply all the transformation in the gene on module `name'"""
		map(lambda x:x.run(name),self.codons)
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

class change:
	"""the change class modelize a set of changes on a module
	it also provide helpers to evaluate the fitness of the module
	basically, it contains an ordered list of genes"""
	def __init__(self,sources,modulename,get_time=20,changes=[]):
		self.sources=sources
		self.modname=modulename
		self.changes=changes
		self.get_time=get_time
		self.foutname=string.join([str(random.Random().randint(0,1000000)) , os.path.basename(self.sources[0])], "_")
		self.execution_time=0.0

	def run(self,testfile,compile=False):
		"""call pips on registered files,
		apply all transformations registered in changes,
		compile the resulting program,
		and run it with  no args in order to get an average execution time"""
		w=simd_workspace(self.sources)
		w.activate("MUST_REGIONS")
		w.activate("PRECONDITIONS_INTER_FULL")
		w.activate("TRANSFORMERS_INTER_FULL")
		w.activate("RICE_SEMANTICS_DEPENDENCE_GRAPH")
		w.set_property(RICEDG_STATISTICS_ALL_ARRAYS=True)
		w.set_property(FOR_TO_DO_LOOP_IN_CONTROLIZER=True)
		c=w[self.modname]
		for t in self.changes:
			t.run(self.modname)
		wdir=self.modname+"/"+self.foutname
		w.save(indir=wdir)
		if compile: #and self.execution_time==0:
			cflags=os.environ["PIPS_CPP_FLAGS"]+" -I. -O0 -march=native -fopenmp "
			exec_out = w.compile(CC="gcc", CFLAGS=cflags,outdir=wdir,outfile="/tmp/a.out",extrafiles=[testfile])
			elapsed=[]
			#print "running",exec_out
			for i in range(0,self.get_time):
				t=-1
				while t<0:
					out=os.popen(exec_out,"r")
					t=out.readline()
					out.close()
					#print "measured:" , t
				elapsed+=[int(t)]
			self.execution_time=elapsed[len(elapsed)/2]
		w.quit()

	def __str__(self):
		s=reduce(lambda x,y:x+" " +y,self.sources,"sources:")
		s+=" module:"+ self.modname+" out:"+self.foutname+"\n"
		s+= "execution time:" + str(self.execution_time) + "\n"
		for t in self.changes:
			s+=str(t)
		return s

#
##
#

def apply_change(source,modulenames,changes,testfile):
	"""apply all changes in ``changes'' with src as seed"""
	the_change=change(source,modulenames,changes=changes)
	#print "running:", the_change
	the_change.run(testfile,compile=True)
	return the_change


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


#
##
#

class full:
	"""perform a full evaluation of the iteration space
	may be vey long, but still usefull as a reference
	for heuristics"""
	def __init__(self,base_genes,sources,module,opts):
		self.base_genes=base_genes
		self.sourcenames=sources
		self.modulename=module
		self.testfile=opts["test"]

	def generate_all_path(self):
		"""generate all `depth'-level combinaison of interaction between elements of `changes'"""
		out=[]
		seed=[[]]
		n=len(self.base_genes)
		for i in range(0,n):
			otmp=[]
			for s in seed:
				for gene in self.base_genes:
					if gene not in s:
						otmp.append(s+[gene])
			print i+1, "out of", n
			seed=otmp
			out+=otmp
		return out

	def run(self):
		"""do the job in three steps:
		# generate all interactions between base genes
		# compute fitness of each element
		# sort result according to fitness"""
		print "generate interactions"
		paths=self.generate_all_path()
		
		print "gather results"
		results=map(lambda x:apply_change(self.sourcenames,self.modulename,x,self.testfile), paths)
		
		print "sort results"
		results.sort(key=operator.attrgetter('execution_time'))
		return results

class greedy:
	"""perform a greedy evaluation of the iteration space:
	at each step, the best elements are selected and
	used as seeds for the next step"""
	def __init__(self,base_genes,sources,module,opts):
		self.base_genes=base_genes
		self.sourcenames=sources
		self.modulename=module
		self.testfile=opts["test"]
		if opts["select"]:self.select=self.select
		else:self.select=3

	def run(self):
		"""do the job"""
		kept=[[]]
		nbiter=len(self.base_genes)
		for i in range(0,nbiter):
			pool=[]
			for k in kept:
				pool.append(k)
				for g in self.base_genes:
					if g not in k:
						pool.append(k+[g])
			results=map(lambda x:apply_change(self.sourcenames,self.modulename,x,self.testfile), pool)
			results.sort(key=operator.attrgetter('execution_time'))
			print i,results[0].execution_time
			kept=map(lambda x:x.changes,results[0:self.select])
		return results

class genetic:
	"""use genetic algorithm to explore the transformation space"""
	def __init__(self,base_genes,sources,module,opts):
		self.base_genes=base_genes
		self.sources=sources
		self.module=module
		self.testfile=opts["test"]
		if opts["nbgen"]:self.nbgen=opts["nbgen"]
		else:self.nbgen=1
		if opts["popsize"]:self.popsize=opts["popsize"]
		else:self.popsize=len(base_genes)
		if opts["nbtournament"]: nbtournament=opts["nbtournament"]
		else:nbtournament=3
		if opts["nbcrossovers"]: nbcrossovers=opts["nbcrossovers"]
		else:nbcrossovers=1
		self.nbtournament=min(nbtournament,self.popsize/3)
		self.nbcrossovers=min(nbcrossovers,self.popsize/3)

	def compute_fitness(self,generation):
		print "computing fitness ..."
		return map(lambda x:apply_change(self.sources,self.module,x,self.testfile), generation)

	def tournament(self,fitness,nb=0):
		"""perform `select' battles betwwen element from `fitness' and 
		return a list of winners"""
		if nb == 0: nb=self.nbtournament
		winners=[]
		print "tournament ...",
		while len(winners) < nb:
			fighters=random_items(fitness,2)
			fighters.sort(key=operator.attrgetter('execution_time'))
			if fighters[0] not in winners:
				winners.append(fighters[0])
		print "adding", str(len(winners)), "winners"
		return winners

	def crossover(self,winners):
		"""perform crossovers betwwen winners of tournaments"""
		crossovers=[]
		print "cross overs ...",
		for i in range(0,self.nbcrossovers):
			xmens=random_items(winners,2)
			fst_gene = random.randint(0,len(xmens[0].changes)-1)
			snd_gene = random.randint(0,len(xmens[1].changes)-1)
			if (xmens[0].changes[fst_gene] not in xmens[1].changes) and (xmens[1].changes[snd_gene] not in xmens[0].changes) :
				cyclope=xmens[0]
				serval=xmens[1]
				laser=cyclope.changes[fst_gene]
				cyclope.changes[fst_gene]=serval.changes[snd_gene]
				serval.changes[snd_gene]=laser
				crossovers.append(cyclope)
				crossovers.append(serval)
		print "adding", str(len(crossovers)), "crossovers"
		return crossovers

	def mutate(self,winners):
		"""perform random mutation on winners"""
		print "mutation ...",
		mutants=winners
		for winner in mutants:
			for mutation in random_items(self.base_genes,3):
				if mutation not in winner.changes:
					n=random.randint(0,1+len(winner.changes))
					winner.changes.insert(n,mutation)
					break
		print "adding", str(len(mutants)), "mutation"
		return mutants

	def add2gen(self,gen,elems):
		"""helper function that add the changes of `elems' to `gen'"""
		map(lambda x: gen.append(x.changes),elems)

	def birth(self,fitness,nb):
		"""create `nb' new elements from `base_genes'"""
		births=[]
		print "births ...",
		births=map(lambda x: change(self.sources,self.module,changes=[x]),random_items(self.base_genes,nb))
		#births=random_items(fitness,nb)
		print "adding", str(len(births)), "births"
		return births

	def run(self):
		"""do the job over nbgen steps"""
		# first generation
		fst_generation=map(lambda x: [x],random_items(self.base_genes,self.popsize))
		fitness=self.compute_fitness(fst_generation)
		fitness.sort(key=operator.attrgetter('execution_time'))

		# evaluate each generation
		for gen in range(0,self.nbgen):
			new_generation=[fitness[0].changes]
			# tournament
			winners=self.tournament(fitness)
			self.add2gen(new_generation,winners)
			# cross overs
			crossovers=self.crossover(winners)
			self.add2gen(new_generation,crossovers)
			# mutation
			mutants=self.mutate(winners)
			self.add2gen(new_generation,mutants)
			# completion du cheptel par des vieux
			births=self.birth(fitness,1)
			self.add2gen(new_generation,births)
			births=self.tournament(fitness,self.popsize-len(new_generation))
			self.add2gen(new_generation,births)
			# fitness
			fitness=self.compute_fitness(new_generation)
			fitness.sort(key=operator.attrgetter('execution_time'))
			print gen,fitness[0].execution_time
		# end
		return fitness

def pick_algo(name, base_genes, sources, module, options):
	return globals()[name](base_genes, sources, module,options)

#
##
#

def main():
	gopts=[ "module=", "sources=", "algo=", "select=", "log=" ,"CPPFLAGS=", "nbgen=", "popsize=", "nbtournament=", "nbcrossovers=", "test="]
	try:
		opts, args = getopt.getopt(sys.argv[1:], "", gopts)
	except getopt.GetoptError, err:
		print str(err) # will print something like "option -a not recognized"
		sys.exit(2)
	options={}
	for g in gopts:
		options[g[0:-1]]=None
	options["log"]=sys.stdout
	options["pips_cpp_flags"]=""

	#
	# parse options
	for o, a in opts:
		if o == "--module":
			options["modulename"]=a
		elif o == "--sources":
			options["sourcenames"]=a.split(",")
		elif o == "--algo":
			options["algo"]=a
		elif o == "--select":
			options["select"]=int(a)
		elif o == "--nbtournament":
			options["nbtournament"]=int(a)
		elif o == "--nbcrossovers":
			options["nbcrossovers"]=int(a)
		elif o == "--log":
			options["log"]=open(a,"w")
		elif o == "--CPPFLAGS":
			options["pips_cpp_flags"]=a
		elif o == "--nbgen":
			options["nbgen"]=int(a)
		elif o == "--popsize":
			options["popsize"]=int(a)
		elif o == "--test":
			options["test"]=a
		else:
			assert False, "unhandled option"
	#
	# verify
	if not (options["modulename"] and options["sourcenames"] and options["algo"]):
		print "both module, source and algo must be set"
		sys.exit(2)
	if not options["test"]:
		print "you must define a test function"
		sys.exit(2)
	#
	# init 
	os.environ["PIPS_CPP_FLAGS"]=options["pips_cpp_flags"]
	print "PIPS_CPP_FLAGS=", os.environ["PIPS_CPP_FLAGS"], "--"
	print "create xp directory"
	if os.path.exists(options["modulename"]):
		shutil.rmtree(options["modulename"])
	os.mkdir(options["modulename"])
	print "copy input file to this dir"
	map(lambda x:shutil.copy(x,options["modulename"]),options["sourcenames"])
	
	# should be loaded elsewhere
	print "define transformations"
	unroll2=gene(property("LOOP_LABEL",'"l100"'),property("UNROLL_RATE","2"), transfo("UNROLL"))
	unroll4=gene(property("LOOP_LABEL",'"l100"'),property("UNROLL_RATE","4"), transfo("UNROLL"))
	unroll8=gene(property("LOOP_LABEL",'"l100"'),property("UNROLL_RATE","8"), transfo("UNROLL"))
	#full_unroll=gene(property("UNROLL_RATE","3"),property("LOOP_LABEL",'"l100"'), transfo("UNROLL"))
	#full_unroll2=gene(property("UNROLL_RATE","3"),property("LOOP_LABEL",'"l200"'), transfo("UNROLL"))
	#inlining=gene(property("INLINING_PURGE_LABELS","FALSE"),transfo("UNFOLDING"))
	peel1=gene(property("LOOP_LABEL", '"l100"'),property("INDEX_SET_SPLITTING_BOUND",'"256"'), transfo("INDEX_SET_SPLITTING"))
	#interchange=gene(property("LOOP_LABEL",'"l300"'), transfo("LOOP_INTERCHANGE"))
	#normalize=gene(transfo("LOOP_NORMALIZE"))
	stripmine=gene(property("LOOP_LABEL", '"l100"'), property("STRIP_MINE_KIND", '0'), property("STRIP_MINE_FACTOR","256"),transfo("STRIP_MINE"),transfo("PARTIAL_EVAL"))
	partial_eval=gene(transfo("PARTIAL_EVAL"))
	dce=gene(transfo("SUPPRESS_DEAD_CODE"))
	omp=gene(transfo("COARSE_GRAIN_PARALLELIZATION"))
	#a2p=gene(transfo("ARRAY_TO_POINTER"))
	slc=gene(transfo("SIMD_LOOP_CONST_ELIM"))
	icm=gene(transfo("INVARIANT_CODE_MOTION"))
	scal=gene(transfo("SCALARIZATION"))
	sac=gene(property("SIMD_AUTO_UNROLL_MINIMIZE_UNROLL","FALSE"),property("SIMD_AUTO_UNROLL_SIMPLE_CALCULATION","FALSE"),property("SIMD_FORTRAN_MEM_ORGANISATION","FALSE"),property("SAC_SIMD_REGISTER_WIDTH","128"),transfo("PARTIAL_EVAL"),transfo("SIMD_ATOMIZER"),transfo("SIMDIZER_AUTO_UNROLL"),transfo("PARTIAL_EVAL"),transfo("SUPPRESS_DEAD_CODE"),transfo("CUMULATED_REDUCTIONS"),transfo("SIMD_REMOVE_REDUCTIONS"),transfo("SINGLE_ASSIGNMENT"),transfo("PARTIAL_EVAL"),transfo("SIMDIZER"))
	
#	base_genes=[scal,sac,slc,peel1,icm,omp,unroll2]#,dce,partial_eval,stripmine,peel1,unroll4,unroll8]
	base_genes=[unroll2]#,dce,partial_eval,stripmine,peel1,unroll4,unroll8]

	#
	# launch algo
	results=[]
	# full transversal
	algo=pick_algo(options["algo"],base_genes,options["sourcenames"],options["modulename"],options)
	results=algo.run()

	# output
	results.sort(key=operator.attrgetter('execution_time'))
	print >> options["log"] , "-- best results" 
	for r in results:
		print >> options["log"], r 
		print >> options["log"], "--"
	print "done !"

	

if __name__ == "__main__":
	main()






