#!/usr/bin/env python
from pyps import *
import re # we are gonne use regular expression
import random
import string
import progressbar as pb
import timeit
import sys
from sys import argv
from exceptions import Exception
import os
import shutil

# helpers
def fact(n): return reduce(lambda x,y:x*y, range(1,n+1))


""" find all loop labels in a module object"""
def find_labels(module):
	label_re = re.compile("^ *(\w+): *for")
	lines=module.code()
	labels=[]
	for line in lines:
		labels+=label_re.findall(line)
	
	#if labels:
	#	print "found labels:"
	#	for l in labels: print l
	return labels

""" get a random item from an array """
def random_item(arr):
	n = len(arr) -1
	return arr[random.randint(0,n)]


"""gather int variable and constant"""
def find_integers(module):
	integers_re1 = re.compile("[\[ \=\,\;](\d+)[\] \=\,\;]")
	integers_re2 = re.compile("int\s+(\w+)")
	lines=module.code()
	integers=[] # default to these
	for line in lines:
		integers+=integers_re1.findall(line)
		integers+=integers_re2.findall(line)
	
	integers = list(set(integers))
	#if integers:
	#	print "found integers:"
	#	for i in integers: print i
	return integers

#end helpers
def peel_once(input_file,modulename, labelname, bound, peel_before, get_time=1, compileit=True):
	""" perform a particular peeling"""
	create(input_file)
	c=modules[modulename]
	set_property("FOR_TO_DO_LOOP_IN_CONTROLIZER",True)
	#c.display()
	set_property("LOOP_PEELING_LOOP_LABEL",labelname)
	set_property("LOOP_PEELING_BOUND",bound)
	set_property("LOOP_PEELING_PEEL_BEFORE_BOUND",peel_before)
	c.loop_peeling()
	foutname=string.join([labelname,bound,str(peel_before),os.path.basename(input_file)], "_")
	c.save(foutname)
	if get_time>0 or compileit:
		exec_out = c.compile(CFLAGS="-O3 -march=native",LDFLAGS="-lm")
	if get_time>0:
	#	print >> sys.stderr, "begin timer"
		stmt='os.system("./'+exec_out+'")'
		timer=timeit.Timer(stmt, 'import os');
		s=timer.repeat(get_time,1)
	#	print >> sys.stderr, "end timer"
		execution_time=reduce(lambda x,y:x+y,s)/len(s)
	#	print >> sys.stderr, "etime" , c.execution_time
	else:execution_time=0.0

	close()

	return (foutname,(modulename,labelname,bound,peel_before,execution_time))

def peel_all(input_file, modulename,nbgen):
	# create xp directory
	if os.path.exists(modulename):
		shutil.rmtree(modulename)
	os.mkdir(modulename)
	# copy input file to this dir
	shutil.copy(input_file,modulename)
	# cd
	os.chdir(modulename)

	""" browse all available peeling"""
	create(input_file)
	set_property("FOR_TO_DO_LOOP_IN_CONTROLIZER",True)
	c=modules[modulename]
	bounds=find_integers(c)
	labels=find_labels(c)
	close()
	infiles=[ input_file ]
	results=[]
	for gen in range(1,nbgen+1):
		pbar = pb.ProgressBar(widgets=[str(gen)+' generation: ',pb.Bar()],\
				maxval=len(infiles)*((fact(len(labels))/fact(len(labels) -1))*(2*len(bounds))),\
				fd=sys.stdout)
		pbar.start()
		j=0;
		outfiles=[]
		for ifile in infiles:
			# get labels
			create(input_file)
			set_property("FOR_TO_DO_LOOP_IN_CONTROLIZER",True)
			c=modules[modulename]
			labels=find_labels(c)
			close()
			# next generation
			for bound in bounds:
				for label in labels:
					for peel_before in [True, False]:
						res=peel_once(ifile,modulename,label,bound,peel_before)
						outfiles.append(res[0])
						results.append(res)
						j+=1
						pbar.update(j)
		pbar.finish()
		infiles=outfiles

	return results 


if len(argv) < 3:
	raise Exception("not enough args")

modulename=argv[1]
sourcename=argv[2]
nbgen=1
if len(argv) > 2:
	nbgen = int(argv[3])




#peel_once(sourcename,modulename,"l0","k",True)

results=peel_all(sourcename,modulename,nbgen)
for r in results:
	print  r[0], "with" , r[1][0], r[1][1], r[1][2], r[1][3], "gave" , r[1][4]

#for n in range(0,1):
#	print "**********************"
#	oofiles=[]
#	for f in outfiles:
#		oofiles+=peel_all(modulename+"/"+f,modulename)
#	outfiles=oofiles
#	oofiles=[]




#seed="carto.c"
#create(seed)
#set_property("PREPROCESSOR_MISSING_FILE_HANDLING","generate")
#set_property("FOR_TO_DO_LOOP_IN_CONTROLIZER",True)
#d=modules["dist"]
#c=modules["carto"]
##c.display()
##d.inline()
##c.display()
#
#for i in range(1,10):
#	labels=find_labels(c);
#	lbl=random_item(labels)
#	i=random_item(find_integers(c))
#	print "setting label to" , lbl , "and bound to", i
#	set_property("LOOP_PEELING_LOOP_LABEL",lbl)
#	set_property("LOOP_PEELING_BOUND",i)
#	c.apply("loop_peeling")
#	c.display()
#	c.apply("suppress_dead_code")
#	c.display()
#
##d.inline()
##c.display()
#set_property("LOOP_PEELING_BOUND","1")
#c.apply("loop_peeling")
#c.display()
##c.save()
#close()
