import pyps
import sac
from subprocess import *
from optparse import OptionParser

parser = OptionParser(usage = "%prog -f FUCNTION src1.c src2.c ...",
                      epilog = "Try `$0 -f dotprod .../validation/SAC/kernels/DOTPROD/DOTPROD.c'.")

parser.add_option("-f", "--function", dest = "function",
                  help = "function to optimize")
(opts, sources) = parser.parse_args()

if not opts.function:
    print "The -f argument is mandatory"
    exit(2)

ws = sac.workspace_sac(sources, parent = pyps)
ws.set_property(ABORT_ON_USER_ERROR = True)

print "Initial code"
module = ws[opts.function]
print "Module", module.name, "selected"
module.display()

# Magie !
module.sac()

print "simdized code"
module.display()

def getout(*cmd):
    return Popen(cmd, stdout=PIPE).communicate()[0]

# get the result from the initial, reference file, without SIMD'izing anything
wsname = ws.name
call(["cc"] + sources + ["-o", "%s.database/Tmp/ref" % wsname]) and exit(1)
ref = getout("./%s.database/Tmp/ref" % wsname)

ws.sac_compile(outfile = "%s.database/Tmp/seq" % (wsname),
               outdir =  "%s.database/Tmp" % (wsname))

seq = getout("./%s.database/Tmp/seq" % wsname)

if seq != ref:
    print "seq ko"
    exit(3)
else:
    print "seq ok"

ws.sac_compile_sse(outfile = "%s.database/Tmp/sse" % (wsname),
                   outdir =  "%s.database/Tmp" % (wsname))
sse = getout("./%s.database/Tmp/sse" % wsname)

if sse != ref:
    print "sse ko"
    exit(3)
else:
    print "sse ok"

#ws.close()
