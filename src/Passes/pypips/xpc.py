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
#import progressbar as pb
import time
import sys
from sys import argv
from exceptions import Exception
import os
import shutil
from multiprocessing import Pool

#
##
#

class transfo:
	def __init__(self,transfo):
		self.transfo=transfo
	def __str__(self):return "transformation:"+self.transfo+"\n"
	def run(self,name):
		pypips.apply(self.transfo,name) 
	def __cmp__(self,other):return cmp(self.transfo,other.transfo)

class property:
	def __init__(self,prop, val):
		self.prop=prop
		self.val=val
	def __str__(self):return "property:"+ self.prop+ " value:"+ self.val+"\n"
	def run(self,name):
		pypips.set_property(self.prop,self.val)
	def __cmp__(self,other):
		n=cmp(self.prop,other.prop) 
		if n == 0: return cmp(self.val,other.val) 
		else: return n

class gene:
	def __init__(self,*codons):
		self.codons=codons
	def run(self,name):
		for codon in self.codons:
			codon.run(name)
	def __str__(self):
		s=""
		for codon in self.codons:
			s+=str(codon)
		return s

	def __cmp__(self,other):
		n = cmp(len(self.codons),len(other.codons))
		if n != 0: return n
		else:
			for i in range(0,len(self.codons)):
				n = cmp(self.codons[i],other.codons[i])
				if n != 0: return n
			return 0





class change:
	def __init__(self,source,modulename,get_time=100):
		self.source=source
		self.modname=modulename
		self.changes=[]
		self.get_time=get_time
		self.foutname=string.join([str(random.Random().randint(0,1000000)) , os.path.basename(self.source)], "_")
		self.execution_time=0.0

	def run(self,compile=False):
		#print "extrafile:", extrafile,"--"
		if extrafile:create(self.source,extrafile)
		else :create(self.source)
		set_property(FOR_TO_DO_LOOP_IN_CONTROLIZER=True)
		c=modules[self.modname]
		for t in self.changes:
			t.run(self.modname)
		#c.suppress_dead_code(DEAD_CODE_DISPLAY_STATISTICS=False)
		c.save(self.modname+"/"+self.foutname)
		if compile: #and self.execution_time==0:
			cflags="-DTEST -DSIZE="+size+" -O3 -march=native"
			exec_out = c.compile(CC="gcc", CFLAGS=cflags,LDFLAGS=extrafile,outfile="/tmp/a.out")
			elapsed=0
			for i in range(0,self.get_time):
				t=-1
				while t<0:
					out=os.popen(exec_out,"r")
					t=out.readline()
					out.close()
				elapsed+=int(t)
			self.execution_time=elapsed/self.get_time
		close()
	def __str__(self):
		s= "source:"+ self.source+ " module:"+ self.modname+" out:"+self.foutname+"\n"
		s+= "execution time:" + str(self.execution_time) + "\n"
		for t in self.changes:
			s+=str(t)
		return s

#
##
#

def apply_change(source,modulename,changes):
	"""apply all changes in ``changes'' with src as seed"""
	the_change=change(source,modulename)
	the_change.changes+=changes
	#print "running:", the_change
	the_change.run(compile=True)
	return the_change

def generate_all_path(base_genes):
	"""generate all ``depth''-level combinaison of interaction between elements of ``changes''"""
	out=[]
	seed=[[]]
	n=len(base_genes)
	for i in range(0,n):
		otmp=[]
		for s in seed:
			for gene in base_genes:
				if gene not in s:
					otmp.append(s+[gene])
		seed=otmp
		out+=otmp
	return out


#
##
#


#""" find all loop labels in a module object"""
#def find_labels(module):
#	label_re = re.compile("^ *(\w+): *for")
#	lines=module.code()
#	labels=[]
#	for line in lines:
#		lbl=label_re.findall(line)
#		if lbl and not (lbl[0] == "l0" or lbl[0] == "l1" ):
#			labels+=lbl
#	
#	#if labels:
#	#	print "found labels:"
#	#	for l in labels: print l
#	return labels
#
def random_items(arr,n):
	""" get a random array from an array """
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
#
#"""gather int variable and constant"""
#def find_integers(module):
#	integers_re1 = re.compile("[\[ \=\,\;](\d+)[\] \=\,\;]")
#	integers_re2 = re.compile("int\s+(\w+)")
#	lines=module.code()
#	integers=["i", "j"] # default to these
#	for line in lines:
#		integers+=integers_re1.findall(line)
#		#integers+=integers_re2.findall(line)
#	
#	integers = list(set(integers))
#	#if integers:
#	#	print "found integers:"
#	#	for i in integers: print i
#	return integers


#def peel_once(other,label,bound,peel_before):
#	def stringify(s): return '"'+s+'"'
#	other.add(property("LOOP_PEELING_LOOP_LABEL",stringify(label)))
#	other.add(property("LOOP_PEELING_BOUND",stringify(bound)))
#	other.add(property("LOOP_PEELING_PEEL_BEFORE_BOUND",upper(str(peel_before))))
#	other.add(transfo("LOOP_PEELING"))
#	other.run(compile=True)
#	other.close()
#	return other
#
#def ppo(params):return peel_once(*params)
#
#def peel_all(input_file, modulename,nbgen):
#	# create xp directory
#	if os.path.exists(modulename):
#		shutil.rmtree(modulename)
#	os.mkdir(modulename)
#	# copy input file to this dir
#	shutil.copy(input_file,modulename)
#
#	""" browse all available peeling"""
#	initial=change(input_file,modulename)
#	initial.run()
#	c=modules[modulename]
#	labels=find_labels(c)
#	bounds=find_integers(c)
#	initial.close()
#	inputs=[initial]
#	outputs=[]
#	for gen in range(1,nbgen+1):
#		#pbar = pb.ProgressBar(widgets=[str(gen)+' generation: ',pb.Bar()],\
#			#	maxval=len(inputs)*((fact(len(labels))/fact(len(labels) -1))*(2*len(bounds))),\
#			#	fd=sys.stdout)
#		#pbar.start()
#		#j=0;
#		outputs=[]
#		for input in inputs:
#			# get labels
#			input.run()
#			c=modules[modulename]
#			labels=find_labels(c)
#			close()
#			# next generation
#			params=[]
#			for bound in bounds:
#				for label in labels:
#					for peel_before in [True, False]:
#						params.append((deepcopy(input),label,bound,peel_before))
#			#noutputs=pool.map(ppo,params)
#			noutputs=map(ppo,params)
#			outputs+=noutputs
#
#						#j+=1
#						#pbar.update(j)
#		#pbar.finish()
#		outputs.sort(key=operator.attrgetter('execution_time'))
#		inputs=outputs[0:select]
#
#	return outputs 

def road_runner(x):return apply_change(x[0],x[1],x[2])

def apply_changes(pool,sourcename, modulename,theset):
	args=[]
	for i in range(0,len(theset)):args.append((sourcename,modulename,theset[i]))
	return pool.map(road_runner,args)
##


size=10
extrafile=""
def main():
	global extrafile,size
	try:
		opts, args = getopt.getopt(sys.argv[1:], "", [ "module=", "source=", "algo=", "select=", "log=" ,"size=", "extrafile=", "nbgen=", "popsize=", "nbproc="])
	except getopt.GetoptError, err:
		print str(err) # will print something like "option -a not recognized"
		sys.exit(2)
	log=sys.stdout
	modulename=None
	sourcename=None
	algo=None
	select=-1
	nbgen=-1
	popsize=-1
	nbproc=1
	#
	# parse options
	for o, a in opts:
		if o == "--module":
			modulename=a
		elif o == "--source":
			sourcename=a
		elif o == "--algo":
			algo=a
		elif o == "--select":
			select=int(a)
		elif o == "--log":
			log=open(a,"w")
		elif o == "--size":
			size=a
		elif o == "--extrafile":
			extrafile=a
		elif o == "--nbgen":
			nbgen=int(a)
		elif o == "--popsize":
			popsize=int(a)
		elif o == "--nbproc":
			nbproc=int(a)
		else:
			assert False, "unhandled option"
	#
	# verify
	if not (modulename and sourcename and algo):
		print "both module, source and algo must be set"
		sys.exit(2)
	#
	# init 
	pips_cpp_flags="-DSIZE="+size+" "
	os.environ["PIPS_CPP_FLAGS"]=pips_cpp_flags
	print "PIPS_CPP_FLAGS=", os.environ["PIPS_CPP_FLAGS"], "--"
	print "create xp directory"
	if os.path.exists(modulename):
		shutil.rmtree(modulename)
	os.mkdir(modulename)
	#ppool = Pool(processes=nbproc)
	
	print "copy input file to this dir"
	shutil.copy(sourcename,modulename)
	
	print "define transformations"
	#unroll2=gene(property("LOOP_LABEL",'"l1"'),property("UNROLL_RATE","2"), transfo("UNROLL"))
	#unroll4=gene(property("LOOP_LABEL",'"l1"'),property("UNROLL_RATE","4"), transfo("UNROLL"))
	fullunroll=gene(property("LOOP_LABEL",'"l1"'), transfo("FULL_UNROLL"))
	inlining=gene(transfo("UNFOLDING"))
	peel1=gene(property("LOOP_LABEL", '"l0"'),property("LOOP_PEELING_BOUND",'"0"'), transfo("LOOP_PEELING"))
	interchange=gene(property("LOOP_LABEL",'"l0"'), transfo("LOOP_INTERCHANGE"))
	
	base_genes=[fullunroll, inlining, peel1, interchange]
	#
	# launch algo
	results=[]
	# full transversal
	if algo == "full":
		print "generate interactions"
		paths=generate_all_path(base_genes)
		
		#for p in paths: print "-----\n",p,"\n-----\n"
		
		print "gather results"
		results=map(lambda x:apply_change(sourcename,modulename,x), paths)
		#results=apply_changes(ppool,sourcename,modulename,paths)
		
		print "sort results"
		results.sort(key=operator.attrgetter('execution_time'))
	# greedy selection
	elif algo== "greedy":
		if select <= 0:
			print "greedy algo requires select> 0"
			sys.exit(2)
		kept=[[]] #map(lambda x:[x],base_genes)
		nbiter=len(base_genes)
		for i in range(0,nbiter):
			pool=[]
			for k in kept:
				pool.append(k)
				for g in base_genes:
					if g not in k:
						pool.append(k+[deepcopy(g)])
			results=map(lambda x:apply_change(sourcename,modulename,x), pool)
			results.sort(key=operator.attrgetter('execution_time'))
			kept=map(lambda x:x.changes,results[0:select])
	# genetic selection
	elif algo=="genetic":
		if nbgen <=0 or select <=0:
			print "genetic algo requires nbgen>0 and select > 0"
			print "got", str(nbgen), "and", str(select)
			sys.exit(2)
		# generate
		if popsize <= 0:popsize=len(base_genes)
		#popsize=min(popsize,len(base_genes))
		select=min(select,popsize/2)
		generation=deepcopy(map(lambda x: [x],random_items(base_genes,popsize)))
		# fitness
		print "computing fitness ..."
		fitness=map(lambda x:apply_change(sourcename,modulename,x), generation)
		for gen in range(0,nbgen):
			generation=[]
			# tournament
			print "tournament ...",
			winners=[]
			while len(winners) < select:
				fighters=random_items(fitness,2)
				fighters.sort(key=operator.attrgetter('execution_time'))
				if fighters[0].changes not in winners:winners.append(fighters[0].changes)
			print "adding", str(len(winners)), "winners"
			generation+=deepcopy(winners)

			# mutation
			print "mutation ...",
			for winner in winners:
				for mutation in random_items(base_genes,2):
					if mutation not in winner:
						winner.append(mutation)
						break
			print "adding", str(len(winners)), "mutation"
			generation+=deepcopy(winners)
			# completion du cheptel
			print "births ...",
			births=map(lambda x: [x],random_items(base_genes,popsize-len(generation)))
			print "adding", str(len(births)), "births"
			generation+=deepcopy(births)
			# fitness
			print "computing fitness ..."
			fitness=map(lambda x:apply_change(sourcename,modulename,x), generation)
		# end
		results=fitness



	else:
		print "unknow algo:" , algo, "--"
		sys.exit(2)

	# output
	results.sort(key=operator.attrgetter('execution_time'))
	print >> log , "-- best results" 
	for r in results[0:select]:
		print >> log , r 
	print >> log , "-- other results" 
	for r in results[select:]:
		print >> log , r 
	print "done !"

	

if __name__ == "__main__":
	main()






