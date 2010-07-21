from pyps import *
from subprocess import *

modulename = "DOTPROD"
wsname = "dotprod_c"

sources = ["kernels/%s/%s.c" % (modulename, modulename), "include/SIMD.c"]
ws = workspace(sources, name = wsname)
ws.set_property(ABORT_ON_USER_ERROR = True)

dotprod = ws['dotprod']

print "initial code"
dotprod.display()

print "simdized code"
dotprod.split_update_operator()
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
