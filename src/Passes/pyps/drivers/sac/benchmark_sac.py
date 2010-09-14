#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pyps
import workspace_gettime as gt
import memalign
import sac

from subprocess import *
import re

benchmarkruns = [
    # {'sources': ["bench/DOTPROD.c"], 'module': "dotprod"},
    # {'sources': ["bench/DOTPROD2.c"], 'module': "dotprod", 'args': "20000000"},
    # {'sources': ["bench/DOTPROD2.c"], 'module': "dotprod",
    #  'unfold': True, 'args': "20000000"},
    # {'sources': ["bench/alphablending.c", "bench/alphablending_main.c"],
    #  'module': "alphablending", 'args': "20000000"},
    # jacobi.c disabled: alignment issues ?
    # {'sources': ["bench/jacobi.c"], 'module': "compute", 'args': "bench/bonjour.pgm"},
    {'sources': ["bench/convol3x3.c"], 'module': "convol"},
    {'sources': ["bench/average_power.c"], 'module': "average_power"},
    # {'sources': ["bench/whetstone.c"], 'module': "main", 'EXTRACFLAGS': "-lm"},
    ]

n_iterations = 1
benchtimes = {}

def gettime(*cmd):
    runtimes = []
    for i in range(0, n_iterations):
        p = Popen(cmd, stdout = PIPE, stderr = PIPE)
        out = p.stdout.read()
        err = p.stderr.read()
        rc = p.wait()
        m = re.search(r"^time for .*: (\d+)$", err)
        if not m:
            print "cmd:", cmd
            print "out:", out
            print "err:", err
            print "rc:", rc
            exit(5)
        time = int(m.group(1))
        runtimes += [time]
    avg = sum(runtimes) / len(runtimes)
    return avg

def benchrun(bench):
    benchname = bench["sources"][0]
    benchtime = {}
    if bench.get("unfold"):
        benchname += "-unfold"
    args = bench.get("args", "")

    ws = pyps.workspace(bench["sources"],
                        parents = [gt.workspace, sac.workspace, memalign.workspace])
    ws.set_property(ABORT_ON_USER_ERROR = True)

    # get the result from the initial, reference file, without SIMD'izing anything
    wsname = ws.name
    ws.compile(outfile = "%s.database/Tmp/ref" % wsname,
               outdir =  "%s.database/Tmp" % wsname,
               CFLAGS = "-O0")
    benchtime["ref"] = gettime("./%s.database/Tmp/ref" % wsname, args)

    ws.compile(outfile = "%s.database/Tmp/ref-O3" % wsname,
               outdir =  "%s.database/Tmp" % wsname,
               CFLAGS = "-O3")
    benchtime["refO3"] = gettime("./%s.database/Tmp/ref-O3" % wsname, args)


    module = ws[bench["module"]]
    # Magie !
    if bench.get("unfold"):
        module.unfolding()

    module.sac(128)

    # Compile using the naïve implementation of SIMD operations
    ws.compile(outfile = "%s.database/Tmp/seq" % (wsname),
               outdir =  "%s.database/Tmp" % (wsname))

    benchtime["seq"] = gettime("./%s.database/Tmp/seq" % wsname, args)

    # Replace the SIMD_* functions with SSE ones. Compile once with
    # ICC, once with GCC.
    ws.sse_compile(outfile = "%s.database/Tmp/sse" % (wsname),
                   outdir =  "%s.database/Tmp" % (wsname))
    benchtime["sse"] = gettime("./%s.database/Tmp/sse" % wsname, args)

    ws.close()

    benchtimes[benchname] = benchtime

for bench in benchmarkruns:
    benchrun(bench)

columns = ["ref", "refO3", "seq", "sse"]
print "\\",
for col in columns:
    print col,
print
for benchname, benchtime in benchtimes.iteritems():
    print benchname,
    for col in columns:
        print benchtime[col], 
    print
