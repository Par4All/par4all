#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pyrops
import workspace_gettime as gt
import memalign
import sac
import sys
import os
import time

from subprocess import *
from optparse import OptionParser
import re

benchmarkruns = [
    # {'sources': ["bsdcos.c"], "module": "__kernel_cos"},
    # memalign for jacobi.c disabled: alignment issues
    {'sources': ["jacobi.c"], 'module': "compute", 'args': "bonjour.pgm"},
    {'sources': ["DOTPROD.c"], 'module': "dotprod"},
    {'sources': ["DOTPROD2.c"], 'module': "dotprod", 'args': "20000000"},
    {'sources': ["DOTPROD2.c"], 'module': "dotprod", 'memalign': True,
     'unfold': True, 'args': "20000000"},
    {'sources': ["alphablending.c", "alphablending_main.c"],
     'module': "alphablending", 'args': "20000000", 'memalign': True},
    {'sources': ["convol3x3.c"], 'module': "convol"},
    {'sources': ["convol3x3.c"], 'module': "convol",
     'name': "convol3x3-unroll", 'path': {'full_unroll_step': 3}},
    {'sources': ["average_power.c"], 'module': "average_power"},
    {'sources': ["whetstone.c"], 'module': "main", 'EXTRACFLAGS': "-lm",
     'cppflags': "-DWITH_TRIGO", 'path': {'if_conversion': True}},
    ]

n_iterations = 50

parser = OptionParser(usage = "%prog")
parser.add_option("-s", "--strict", dest = "strict",
                  action = "store_true", default = False,
                  help = "check program output")
parser.add_option("-q", "--quick", dest = "quick",
                  action = "store_true", default = False,
                  help = "do only one iteration when timing stuff")
parser.add_option("--cflags", "--CFLAGS", dest = "cflags",
                  help = "additionnal CFLAGS for all compilations")
parser.add_option("--outfile", dest = "outfile",
                  help = "put the results in a file suitable for gnuplot")
parser.add_option("--normalize", dest = "normalize", action = "store_true",
                  default = False, help = "normalize timing results")
(opts, _) = parser.parse_args()

benchtimes = {}
errors = []

if opts.quick:
    for bench in benchmarkruns:
        if "args" in bench and re.match("\d+", bench["args"]):
            bench["args"] = "200"
    n_iterations = 2

def benchrun(bench):
    benchtime = {}
    referenceout = []
    def tryBenchmark(*args, **kwargs):
        kwargs["args"]       = [ bench.get("args", "") ]
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
        EXTRACFLAGS += opts.cflags

    parents = [gt.workspace, sac.workspace]
    if bench.get("memalign"):
        parents.append(memalign.workspace)
    ws = pyrops.pworkspace(bench["sources"],
                           parents = parents,
                           cppflags = bench.get("cppflags", ""))

    # get the result from the initial, reference file, without SIMD'izing anything
    benchtime["gcc"] = tryBenchmark("gcc", CFLAGS = "-O3 " + EXTRACFLAGS)
    benchtime["icc"] = tryBenchmark("icc", CC = "icc",
                                    CFLAGS = "-O3 " + EXTRACFLAGS)

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

    if bench.get("memalign"):
        # Try with alignement
        ws.memalign()

    # Compile using the naïve implementation of SIMD operations
    benchtime["gcc+seq"] = tryBenchmark("gcc+seq", CFLAGS = "-O3 " + EXTRACFLAGS)

    # Replace the SIMD_* functions with SSE ones. Compile once with
    # ICC, once with GCC. Note that simd_compile always uses -O3.
    benchtime["gcc+sse"] = tryBenchmark("gcc+sse", compilemethod = ws.simd_compile,
                                    CFLAGS = EXTRACFLAGS)
    benchtime["icc+sse"] = tryBenchmark("icc+sse", compilemethod = ws.simd_compile,
                                        CC = "icc", CFLAGS = EXTRACFLAGS)

    #ws.close()

    benchtimes[bench['name']] = benchtime

for bench in benchmarkruns:
    benchrun(bench)

if errors != []:
    print "There were some errors:"
for e in errors:
    print e

columns = ["gcc", "icc", "gcc+seq", "gcc+sse", "icc+sse"]

print benchtimes

if opts.outfile:
    outfile = open(opts.oufile, "w")
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
            time = time / float(benchtime.get("gcc", 1))
        outfile.write(str(time))
        outfile.write("\t")
    outfile.write("\n")

if errors != []:
    exit(1)
