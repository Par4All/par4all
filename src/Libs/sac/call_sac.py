import sac
import pyps
from subprocess import *
import sys
import re

sources = sys.argv[1:]

ws = sac.sac_workspace(sources)
wsname = ws.name
ws.set_property(ABORT_ON_USER_ERROR = True)

print "Initial code"
dotprod = ws['dotprod']
print "Module dotprod selected"
dotprod.display()

dotprod.sac()

print "simdized code"
dotprod.display()

dotprod.unsplit()

def getout(*cmd):
    return Popen(cmd, stdout=PIPE).communicate()[0]

# get the result from the initial, reference file, without SIMD'izing anything
call(["cc"] + sources + ["-o", "%s.database/Tmp/ref" % wsname]) and exit(1)
ref = getout("./%s.database/Tmp/ref" % wsname)

def unincludeSIMD(fname):
    # in the modulename.c file, undo the inclusion of SIMD.h by deleting
    # everything up to the definition of our function (not as clean as could
    # be, to say the least...)
    f = open(fname, "r")
    while not re.search("dotprod", f.readline()):
        pass
    contents = f.readlines()
    f.close()
    f = open(fname, "w")
    f.writelines(contents)
    f.close()

def addBeginning(fname, *args):
    contents = map((lambda(s): s + "\n" if s[-1] != "\n" else s),
                   args)
    
    f = open(fname, "r")
    contents += f.readlines()
    f.close()
    f = open(fname, "w")
    f.writelines(contents)
    f.close()

def reincludeSIMD(fname):
    addBeginning(fname, '#include "SIMD.h"')

def reincludestdio(fname):
    addBeginning(fname, "#include <stdio.h>")

def goingToRunWithFactory(*funs):
    def goingToRunWithAux(s, files, outdir):
        for fname in files:
            if re.search(r"SIMD\.c$", fname):
                continue
            for fun in funs:
                fun(fname)
    return goingToRunWithAux

# compile, undoing the inclusion of SIMD.h
pyps.workspace.goingToRunWith = goingToRunWithFactory(unincludeSIMD,
                                                      reincludeSIMD,
                                                      reincludestdio)
ws.compile(outfile = "%s.database/Tmp/seq" % (wsname),
           outdir =  "%s.database/Tmp" % (wsname),
           CFLAGS = "-Iinclude")
seq = getout("./%s.database/Tmp/seq" % wsname)

if seq != ref:
    print "seq ko"
    exit(3)
else:
    print "seq ok"

def addSSE(fname):
    contents = open("include/sse.h").readlines()
    f = open(fname)
    for line in f:
        line = re.sub(r"float (v4sf_[^[]+)", r"__m128 \1", line)
        line = re.sub(r"float (v4si_[^[]+)", r"__m128i \1", line)
        line = re.sub(r"v4s[if]_([^,[]+)\[[^]]*\]", r"\1", line)
        line = re.sub(r"v4s[if]_([^ ,[]+)", r"\1", line)
        line = re.sub(r"double (v2df_[^[]+)", r"__m128d \1", line)
        line = re.sub(r"double (v2di_[^[]+)", r"__m128i \1", line)
        line = re.sub(r"v2d[if]_([^,[]+)\[[^]]*\]", r"\1", line)
        line = re.sub(r"v2d[if]_([^ ,[]+)", r"\1", line)
        contents.append(line)
    f.close()
    f = open(fname, "w")
    f.writelines(contents)
    f.close()

# recompile, using sse.h instead of SIMD.h
pyps.workspace.goingToRunWith = goingToRunWithFactory(unincludeSIMD,
                                                      reincludestdio,
                                                      addSSE)

ws.compile(outfile = "%s.database/Tmp/sse" % (wsname),
           outdir =  "%s.database/Tmp" % (wsname),
           CFLAGS = "-Iinclude")
sse = getout("./%s.database/Tmp/sse" % wsname)

if sse != ref:
    print "sse ko"
    exit(3)
else:
    print "sse ok"

ws.close()
