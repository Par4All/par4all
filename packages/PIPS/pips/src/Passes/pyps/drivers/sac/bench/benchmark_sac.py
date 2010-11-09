#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import with_statement # to cope with python2.5

import pyrops
import workspace_gettime as gt
import memalign
import sac
import sys
import os
import time
import shutil

from subprocess import *
from optparse import OptionParser
import re

benchmarkruns = [
	## memalign for jacobi.c disabled: alignment issues
	#{'sources': ["jacobi.c"], 'module': "compute", 'args': "bonjour.pgm"},
	#{'sources': ["ddot_r.c"], 'module': "ddot_r",  'memalign': True, 'args':"2000000"},
	#{'sources': ["ddot_ur.c"], 'module': "ddot_ur",  'memalign': True, 'args':"2000000"},
	#{'sources': ["daxpy_r.c"], 'module': "daxpy_r",  'memalign': True, 'args':"2000000"},
	#{'sources': ["daxpy_ur.c"], 'module': "daxpy_ur",  'memalign': True, 'args':"2000000"},
	#{'sources': ["dscal_r.c"], 'module': "dscal_r",  'memalign': True, 'args':"2000000"},
	#{'sources': ["dscal_ur.c"], 'module': "dscal_ur",  'memalign': True, 'args':"2000000"},
	{'sources': ["alphablending.c", "alphablending_main.c"],
	 'module': "alphablending", 'args': "20000000", 'memalign': True},
	#{'sources': ["fir.c"], 'module': "FIRFilter",'args':"100000", 'memalign': False},
	#{'sources': ["convol3x3.c"], 'module': "convol", 'args':'256'},
	#{'sources': ["average_power.c"], 'module': "average_power"},
	#{'sources':["matrix_mul_const.c"], 'module':"matrix_mul_const", 'args':'1000', 'memalign': True },
	#{'sources':["matrix_add_const.c"], 'module':"matrix_add_const", 'args':'1000', 'memalign': True },
	#{'sources':["matrix_mul_vect.c"], 'module':"matrix_mul_vect", 'args':'1000', 'memalign': True ,'path': { 'enhanced_reduction':True}},
	#{'sources':["matrix_mul_matrix.c"], 'module':"matrix_mul_matrix", 'args':'1000', 'memalign': True ,'path': { 'enhanced_reduction':True}},
	]

n_iterations = 50

parser = OptionParser(usage = "%prog")
parser.add_option("-s", "--strict", dest = "strict",
				  action = "store_true", default = False,
				  help = "check program output")
parser.add_option("-q", "--quick", dest = "quick",
				  action = "store_true", default = False,
				  help = "do only one iteration when timing stuff")
parser.add_option("--blork", dest = "blork",
				  action = "store_true", default = False,
				  help = "do not destroy workspace on close")
parser.add_option("--cflags", "--CFLAGS", dest = "cflags",
				  help = "additionnal CFLAGS for all compilations")
parser.add_option("--outfile", dest = "outfile",
				  help = "put the results in a file suitable for gnuplot")
parser.add_option("--normalize", dest = "normalize", action = "store_true",
				  default = False, help = "normalize timing results")
parser.add_option("--driver", dest = "driver", default = "sse",
				  help = "3DNow or SSE (the default)")
parser.add_option("--outdir", dest = "outdir",
				  help = "put the resulting transformation in this directory")
(opts, _) = parser.parse_args()

benchtimes = {}
errors = []

if opts.quick:
	for bench in benchmarkruns:
		if "args" in bench and re.match("\d+", bench["args"]):
			bench["args"] = "200"
	n_iterations = 1

if opts.outdir:
	try: shutil.rmtree(opts.outdir)
	except: pass
	os.makedirs(opts.outdir)

def benchrun(bench):
	benchtime = {}
	referenceout = []
	def tryBenchmark(*args, **kwargs):
		if bench.get("args",None):
			kwargs["args"]	   = [ bench.get("args",None) ]
		kwargs["iterations"] = n_iterations
		kwargs["reference"]  = (referenceout if opts.strict else False)
		time = 0
		try:
			time = ws.benchmark(*args, **kwargs)
		except RuntimeError, e:
			errors.append(e)
			if opts.strict: raise
		return time

	# build a name for the benchmark using the basename of the first source file
	if not bench.has_key("name"):
		bench["name"] = os.path.basename(bench["sources"][0])
		if bench.get("unfold"):
			bench["name"] += "-unfold"

	EXTRACFLAGS = bench.get("EXTRACFLAGS", "")
	if opts.cflags:
		EXTRACFLAGS += " " + opts.cflags

	parents = [gt.workspace, sac.workspace]
	if bench.get("memalign",False):
		parents.append(memalign.workspace)
	with pyrops.pworkspace(bench["sources"],
						   parents = parents,
						   deleteOnClose = not opts.blork,
						   driver = opts.driver,
						   cppflags = bench.get("cppflags", "")) as ws:

		# get the result from the initial, reference file, without SIMD'izing anything
		if opts.outdir:
			ws.compile(rep = "%s/ref/%s" % (opts.outdir, bench["name"]),
					   CFLAGS = "-O3 " + EXTRACFLAGS)
		else:
			benchtime["gcc-ref"] = tryBenchmark("gcc", CFLAGS = "-O3 -fno-tree-vectorize" + EXTRACFLAGS)
			benchtime["gcc"] = tryBenchmark("gcc", CFLAGS = "-O3 " + EXTRACFLAGS)
			benchtime["icc"] = tryBenchmark("icc", CC = "icc", CFLAGS = "-O3 " + EXTRACFLAGS)
			benchtime["llvm"] = tryBenchmark("llvm", CC = "llvm-gcc-4.2", CFLAGS = "-O3 " + EXTRACFLAGS)

		module = ws[bench["module"]]
		# Magie !
		if bench.get("unfold"):
			module.unfolding()

		try:
			if bench.get("path"):
				module.sac(**bench["path"])
			else:
				module.sac()
		except RuntimeError, e:
			errors.append(e.args)
			if opts.strict: raise

		if bench.get("memalign",False):
			# Try with alignement
			ws.memalign()

		# Compile using the naïve implementation of SIMD operations
		if opts.outdir:
			ws.compile(rep = "%s/seq/%s" % (opts.outdir, bench["name"]),
					   CFLAGS = "-O3 " + EXTRACFLAGS)
		else:
			benchtime["gcc+seq"] = tryBenchmark("gcc+seq", CFLAGS = "-O3 " + EXTRACFLAGS)

		# Replace the SIMD_* functions with SSE ones. Compile once with
		# ICC, once with GCC. Note that simd_compile always uses -O3.
		if opts.outdir:
			ws.simd_compile(rep = "{0}/{1}/{2}".format(opts.outdir, opts.driver, bench["name"]),
							CFLAGS = EXTRACFLAGS)
		else:
			benchtime["gcc+sac"] = tryBenchmark("gcc+sac", compilemethod = ws.simd_compile,
												CFLAGS = EXTRACFLAGS)
			benchtime["icc+sac"] = tryBenchmark("icc+sac", compilemethod = ws.simd_compile,
												CC = "icc", CFLAGS = EXTRACFLAGS)
			benchtime["llvm+sac"] = tryBenchmark("llvm+sac", compilemethod = ws.simd_compile,
												CC = "llvm-gcc-4.2", CFLAGS = EXTRACFLAGS)

	benchtimes[bench['name']] = benchtime

for bench in benchmarkruns:
	benchrun(bench)

if errors != []:
	print "There were some errors:"
for e in errors:
	print e

# if we only wanted the resulting files, we are done now
if opts.outdir:
	exit(0)
# Otherwise print the results
columns = ["gcc-ref", "gcc", "icc", "llvm", "gcc+seq", "gcc+sac", "icc+sac", "llvm+sac"]

if opts.outfile:
	outfile = open(opts.outfile, "w")
else:
	outfile = sys.stdout
outfile.write("# results for benchmark_sac.py, generated the "+ time.asctime() +"\n")
outfile.write("# Based on " + os.popen('git log|head -n1').read() + "\n")
outfile.write("Compilation\t"+ "\t".join(columns) + "\n")
for benchname, benchtime in benchtimes.iteritems():
	outfile.write(benchname + "\t")
	for col in columns:
		time = benchtime.get(col, "0")
		if opts.normalize:
			if col == 'gcc':
				time = - ( (time / float(benchtime.get("gcc-ref", 1) ) -1 ) *100)
			else:
				time = - ( (time / float(benchtime.get("gcc", 1) ) -1 ) *100)

		outfile.write(str(time))
		outfile.write("\t")
	outfile.write("\n")

if errors != []:
	exit(1)
