import sac
import pyps
from subprocess import *
import sys
import re
from optparse import OptionParser

parser = OptionParser(usage = "%prog -f FUCNTION src1.c src2.c ...",
                      epilog = "Try `$0 -f dotprod .../validation/SAC/kernels/DOTPROD/DOTPROD.c'.")

parser.add_option("-f", "--function", dest = "function",
                  help = "function to optimize")
(opts, sources) = parser.parse_args()

if not opts.function:
    print "The -f argument is mandatory"
    exit(2)

ws = sac.sac_workspace(sources)
wsname = ws.name
ws.set_property(ABORT_ON_USER_ERROR = True)

print "Initial code"
module = ws[opts.function]
print "Module", module.name, "selected"
module.display()

module.sac()

print "simdized code"
module.display()

module.unsplit()

def getout(*cmd):
    return Popen(cmd, stdout=PIPE).communicate()[0]

# get the result from the initial, reference file, without SIMD'izing anything
call(["cc"] + sources + ["-o", "%s.database/Tmp/ref" % wsname]) and exit(1)
ref = getout("./%s.database/Tmp/ref" % wsname)

sac.sac_compile(ws,
                outfile = "%s.database/Tmp/seq" % (wsname),
                outdir =  "%s.database/Tmp" % (wsname),
                CFLAGS = "-I.")

seq = getout("./%s.database/Tmp/seq" % wsname)

if seq != ref:
    print "seq ko"
    exit(3)
else:
    print "seq ok"

sac.sac_compile_sse(ws,
                    outfile = "%s.database/Tmp/sse" % (wsname),
                    outdir =  "%s.database/Tmp" % (wsname),
                    CFLAGS = "-Iinclude")
sse = getout("./%s.database/Tmp/sse" % wsname)

if sse != ref:
    print "sse ko"
    exit(3)
else:
    print "sse ok"

#ws.close()
