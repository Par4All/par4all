from pyps import *
from subprocess import *
import shutil

modulename = "DOTPROD"
wsname = "dotprod_c"

sources = ["kernels/%s/%s.c" % (modulename, modulename), "include/SIMD.c"]
try:
    shutil.rmtree(wsname + ".database")
except:
    pass

ws = workspace(sources, name = wsname)
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

dotprod.display()
print ws.get_property("IF_CONVERSION_INIT_THRESHOLD")
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

system("sed", "-i", "-e", "1,/dotprod/ d",
       "%s.database/Src/%s.c" % (wsname, modulename))
system("cc", "kernels/%s/%s.c" % (modulename, modulename), "include/SIMD.c",
       "-o", "%s.database/Tmp/ref" % wsname)
ref = getout("./%s.database/Tmp/ref" % wsname)

system("sed", "-i", "-e", "1 i #include \"SIMD.h\"",
       "%s.database/Src/%s.c" % (wsname, modulename))
# This wasn't necessary in the .tpips version
system("sed", "-i", "-e", "1 i #include <stdio.h>",
       "%s.database/Src/%s.c" % (wsname, modulename))
system("cc", "-Iinclude", "%s.database/Src/%s.c" % (wsname, modulename),
       "include/SIMD.c", "-o", "%s.database/Tmp/seq" % (wsname))
seq = getout("./%s.database/Tmp/seq" % wsname)

if ref != seq:
    print "seq ko"
else:
    print "seq ok"

# system("./compileC.sh $WS $module.c $WS.database/Tmp/sse.c")
# system("cc -O3 -I. -march=native $WS.database/Tmp/sse.c -o $WS.database/Tmp/sse")
# system("if test "`./$WS.database/Tmp/ref`" = "`$WS.database/Tmp/sse`" ; then echo sse-ok ; else echo sse-ko ; fi"")
