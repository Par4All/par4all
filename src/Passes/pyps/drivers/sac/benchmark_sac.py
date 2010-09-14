#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pyrops
import workspace_gettime as gt
import memalign
import sac
import sys
import os

from subprocess import *
from optparse import OptionParser
import re

benchmarkruns = [
    # {'sources': ["bench/bsdcos.c"], "module": "__kernel_cos"},
    {'sources': ["bench/DOTPROD.c"], 'module': "dotprod"},
    {'sources': ["bench/DOTPROD2.c"], 'module': "dotprod", 'args': "20000000"},
    {'sources': ["bench/DOTPROD2.c"], 'module': "dotprod",
     'unfold': True, 'args': "20000000"},
    {'sources': ["bench/alphablending.c", "bench/alphablending_main.c"],
     'module': "alphablending", 'args': "20000000"},
    # jacobi.c disabled: alignment issues ?
    {'sources': ["bench/jacobi.c"], 'module': "compute", 'args': "bench/bonjour.pgm"},
    {'sources': ["bench/convol3x3.c"], 'module': "convol"},
    {'sources': ["bench/average_power.c"], 'module': "average_power"},
    {'sources': ["bench/whetstone.c"], 'module': "main", 'EXTRACFLAGS': "-lm",
     'cppflags': "-DWITH_TRIGO"},
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
        kwargs["reference"]  = referenceout
        time = "XXXX"
        try:
            time = ws.benchmark(*args, **kwargs)
        except:
            if opts.strict: raise
        return time

    # build a name for the benchmark using the basename of the first source file
    bench["name"] = os.path.basename(bench["sources"][0])
    if bench.get("unfold"):
        bench["name"] += "-unfold"

    EXTRACFLAGS = bench.get("EXTRACFLAGS", "")
    if opts.cflags:
        EXTRACFLAGS += opts.cflags

    ws = pyrops.pworkspace(bench["sources"],
                        parents = [gt.workspace, sac.workspace, memalign.workspace],
                        cppflags = bench.get("cppflags", ""))

    # get the result from the initial, reference file, without SIMD'izing anything
    benchtime["ref"] = tryBenchmark("ref", CFLAGS = "-O0 " + EXTRACFLAGS)
    benchtime["refO3"] = tryBenchmark("refO3", CFLAGS = "-O3 " + EXTRACFLAGS)
    benchtime["refO3-icc"] = tryBenchmark("refO3-icc", CC = "icc",
                                          CFLAGS = "-O3 " + EXTRACFLAGS)

    module = ws[bench["module"]]
    # Magie !
    if bench.get("unfold"):
        module.unfolding()

    try:
        module.sac()
    except:
        if opts.strict: raise

    # Compile using the naïve implementation of SIMD operations
    benchtime["seq"] = tryBenchmark("seq", CFLAGS = "-O3 " + EXTRACFLAGS)

    # Replace the SIMD_* functions with SSE ones. Compile once with
    # ICC, once with GCC. Note that simd_compile always uses -O3.
    benchtime["sse"] = tryBenchmark("sse", compilemethod = ws.simd_compile,
                                    CFLAGS = EXTRACFLAGS)
    benchtime["sse-icc"] = tryBenchmark("sse-icc", compilemethod = ws.simd_compile,
                                        CC = "icc", CFLAGS = EXTRACFLAGS)

    # Try with alignement
    ws.memalign()
    benchtime["sse-align"] = tryBenchmark("sse-aligned",
                                          compilemethod = ws.simd_compile,
                                          CFLAGS = EXTRACFLAGS)
    benchtime["sse-icc-align"] = tryBenchmark("sse-icc-align",
                                              compilemethod = ws.simd_compile,
                                              CC = "icc", CFLAGS = EXTRACFLAGS)

    ws.close()

    benchtimes[bench['name']] = benchtime

for bench in benchmarkruns:
    benchrun(bench)

if errors != []:
    print "their were some errors"
for e in errors:
    print e

columns = ["ref", "refO3", "refO3-icc", "seq", "sse", "sse-icc", "sse-aligned", "sse-icc-aligned"]
print "\\",
for col in columns:
    print col,
print
for benchname, benchtime in benchtimes.iteritems():
    print benchname,
    for col in columns:
        print benchtime.get(col, "XXXX"),
    print

if errors != []:
    exit(1)
