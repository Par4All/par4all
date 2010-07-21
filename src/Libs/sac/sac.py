from pyps import *
from subprocess import *
import shutil

modulename = "DOTPROD"

sources = ["kernels/%s/%s.c" % (modulename, modulename), "include/SIMD.c"]

ws = workspace(sources)
wsname = ws.name
ws.set_property(ABORT_ON_USER_ERROR = True)

print "Initial code"
dotprod = ws['dotprod']
print "Module dotprod selected"
dotprod.display()

dotprod.split_update_operator()

# benchmark.tpips.h begin
ws.activate("MUST_REGIONS")
ws.activate("PRECONDITIONS_INTER_FULL")
ws.activate("TRANSFORMERS_INTER_FULL")

ws.set_property(RICEDG_STATISTICS_ALL_ARRAYS = True)
ws.activate("RICE_SEMANTICS_DEPENDENCE_GRAPH")

ws.set_property(SIMD_FORTRAN_MEM_ORGANISATION = False)
ws.set_property(SAC_SIMD_REGISTER_WIDTH = 128)
ws.set_property(SIMDIZER_AUTO_UNROLL_SIMPLE_CALCULATION = False)
ws.set_property(SIMDIZER_AUTO_UNROLL_MINIMIZE_UNROLL = False)
ws.set_property(PRETTYPRINT_ALL_DECLARATIONS = True)

dotprod.split_update_operator()

dotprod.if_conversion_init()
dotprod.display()

dotprod.if_conversion()
dotprod.display()

dotprod.if_conversion_compact()
#dotprod.use_def_elimination()
dotprod.display()

dotprod.partial_eval()
dotprod.simd_atomizer()
dotprod.display()

dotprod.simdizer_auto_unroll()
dotprod.partial_eval()
dotprod.clean_declarations()
dotprod.suppress_dead_code()
dotprod.display()
#make DOTDG_FILE
dotprod.simd_remove_reductions()
dotprod.display()

#dotprod.deatomizer()
#dotprod.partial_eval()
#dotprod.use_def_elimination()
#dotprod.display()

dotprod.print_dot_dependence_graph()
dotprod.single_assignment()

dotprod.display()

dotprod.simdizer()

dotprod.display()

#dotprod.use_def_elimination()
#dotprod.display()

dotprod.simd_loop_const_elim()
#setproperty EOLE_OPTIMIZATION_STRATEGY "ICM"
#dotprod.optimize_expressions()
#dotprod.partial_redundancy_elimination()
dotprod.display()

#dotprod.use_def_elimination()
dotprod.clean_declarations()
dotprod.suppress_dead_code()
dotprod.display()

# benchmark.tpips.h end

print "simdized code"
dotprod.display()

dotprod.unsplit()

def getout(*cmd):
    return Popen(cmd, stdout=PIPE).communicate()[0]

def system(*cmd):
    if Popen(cmd).wait() != 0:
        exit(1)
    return 0

system("cc", "kernels/%s/%s.c" % (modulename, modulename), "include/SIMD.c",
       "-o", "%s.database/Tmp/ref" % wsname)
ref = getout("./%s.database/Tmp/ref" % wsname)

def unincludeSIMD(fname):
    # in the modulename.c file, undo the inclusion of SIMD.h by deleting
    # everything up to the definition of our function (not as clean as could
    # be, to say the least...)
    f = open(fname, "r")
    print "unincludeSIMD:", fname
    while not re.search("dotprod", f.readline()):
        pass
    print "unincludeSIMD:", "removing dotprod done"
    contents = f.readlines()
    f.close()
    f = open(fname, "w")
    f.writelines(contents)
    f.close()
    print fname

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
                print "goingToRunWithAux:", "arg:", fname
                fun(fname)
    return goingToRunWithAux

workspace.goingToRunWith = goingToRunWithFactory(unincludeSIMD,
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

workspace.goingToRunWith = goingToRunWithFactory(unincludeSIMD,
                                                 reincludestdio,
                                                 addSSE)
#system("./compileC.sh", wsname, modulename+".c", wsname + ".database/Tmp/sse.c")

ws.compile(outfile = "%s.database/Tmp/sse" % (wsname),
           outdir =  "%s.database/Tmp" % (wsname),
           CFLAGS = "-Iinclude")
# system("cc", "-O3", "-I.", "-march=native", wsname + ".database/Tmp/sse.c",
#        "-o", wsname +".database/Tmp/sse")
sse = getout("./%s.database/Tmp/sse" % wsname)

if sse != ref:
    print "sse ko"
    exit(3)
else:
    print "sse ok"

ws.close()
