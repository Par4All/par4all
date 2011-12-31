#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# A sample script showing how to call and use sac.workspace. Also used
# as a test on sac.

import pyps
import sac
import workspace_gettime as gt

from subprocess import *
from optparse import OptionParser
import re
import sys

parser = OptionParser(usage = "%prog [options] -f FUCNTION src1.c src2.c ...",
					  epilog = "Try `$0 -f dotprod .../validation/SAC/kernels/DOTPROD/DOTPROD.c'.")

parser.add_option("-f", "--function", dest = "function", action = "append",
				  help = "function to optimize")
parser.add_option("-a", "--args", dest = "args", action = "append",default = [],
				  help = "arguments to pass to the compiled program")
parser.add_option("-d", "--driver", dest = "driver",
				  default = "sse", help =  "sse or 3dnow")
parser.add_option("-e", "--explore", dest = "explore", default = None,
				  help = "control the running of SAC\n."
				  "The syntax is \"option:True,option2:False\"...")
parser.add_option("-v", "--verbose", dest = "verbose",
				  default = 0, action = "count",
				  help = "verbose output; can be specified several times")
parser.add_option("-s", "--strict", dest = "strict", action = "store_true",
				  help = "check output, and exit when output changes")
parser.add_option("-m", "--memalign", dest = "memalign", default = False,
				  action = "store_true", help = "use memalign.workspace")
parser.add_option("--cppflags", "--CPPFLAGS", dest = "cppflags",
				  default = "", help = "CPP options")
parser.add_option("--cflags", "--CFLAGS", dest = "cflags",
				  default = "", help = "gcc options")
parser.add_option("--ldflags", "--LDFLAGS", dest = "ldflags",
				  default = "", help = "linker options")
parser.add_option("--blork", action = "store_true", default = False,
				  help = "don't clean the workspace on exit")
(opts, sources) = parser.parse_args()

explorationpath = {}
if opts.explore:
	steps = opts.explore.split(",")
	for s in steps:
		(k, v) = s.split(":")
		v = v.lower()
		try:
			explorationpath[k] = int(v)
		except ValueError:
			if v == "true":
				explorationpath[k] = True
			elif v == "false":
				explorationpath[k] = False

if opts.verbose >= 2:
	print explorationpath

if not opts.function:
	print "The -f argument is mandatory"
	exit(2)

parents = [sac.workspace, gt.workspace]
# Run-time composition of workspaces!
if opts.memalign:
	import memalign
	parents.insert(0, memalign.workspace)

ws = pyps.workspace(sources, parents = parents, verbose = (opts.verbose >= 2), driver = opts.driver, cppflags = opts.cppflags)

reference = []
def runws(*args, **kwargs):
	global ws
	kwargs["args"]		 = opts.args
	kwargs["iterations"] = 1
	kwargs["reference"] = (reference if opts.strict else [])
	if "CFLAGS" in kwargs: kwargs["CFLAGS"] += (" " + opts.cflags)
	else: kwargs["CFLAGS"] = opts.cflags
	kwargs["LDFLAGS"] = opts.ldflags
	time = "XXXX"
	try: time = ws.benchmark(*args, **kwargs)
	except RuntimeError, e:
		if opts.strict: raise
		else:
			print >>sys.stderr, e.args
			print >>sys.stderr, "Continuing anyway"
	return time

# get the result from the initial, reference file, without SIMD'izing anything
ref_time = runws("ref", CFLAGS = "-O3")
reficc_time = runws("reficc", CFLAGS = "-O3", CC = "icc")

for f in opts.function:
	module = ws[f]
	if opts.verbose >= 1:
		print "Module", module.name, "selected"
		print "Initial code"
		module.display()

	# Magie !
	try:
		module.sac(verbose = (opts.verbose >= 3), **explorationpath)
	except RuntimeError, e:
		print >>sys.stderr, "Couldn't apply sac on module", f
		print >>sys.stderr, e.args
		if opts.strict:
			module.display()
			raise

	if opts.verbose >= 1:
		print "Simdized code"
		module.display()

# Compile using the sequential (naïve) versions of SIMD instructions
seq_time = runws("seq", CFLAGS = "-O3")

if opts.memalign:
	ws.memalign()
	if opts.verbose >= 1:
		module.display()

sse_time = runws("sse", compilemethod = ws.simd_compile)
icc_time = runws("icc", compilemethod = ws.simd_compile, CC = "icc")

if not opts.blork:
	ws.close()

print "Run times: (ref, reficc, seq, sse, icc):"
print ref_time
print reficc_time
print seq_time
print sse_time
print icc_time
