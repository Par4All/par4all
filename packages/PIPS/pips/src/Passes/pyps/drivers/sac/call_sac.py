#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# A sample script showing how to call and use sac.workspace. Also used
# as a test on sac.

import pyps
import sac

from subprocess import *
from optparse import OptionParser
import re

parser = OptionParser(usage = "%prog -f FUCNTION src1.c src2.c ...",
					  epilog = "Try `$0 -f dotprod .../validation/SAC/kernels/DOTPROD/DOTPROD.c'.")

parser.add_option("-f", "--function", dest = "function",
				  help = "function to optimize")
parser.add_option("-a", "--args", dest = "args",
				  help = "arguments to pass to the compiled program")
parser.add_option("-t", "--time", dest = "time",
                  action = "store_true", default = False,
				  help = "use workspace_gettime to time the ran program")
(opts, sources) = parser.parse_args()

if not opts.function:
	print "The -f argument is mandatory"
	exit(2)

# Run-time composition of workspaces!
if opts.time:
	import workspace_gettime as gt
	ws = pyps.workspace(sources, parents = [gt.workspace, sac.workspace])
else:
	ws = pyps.workspace(sources, parents = [sac.workspace])
	
ws.set_property(ABORT_ON_USER_ERROR = True)

def getout(*cmd):
	if opts.args:
		cmd += (opts.args,)
	p = Popen(cmd, stdout = PIPE, stderr = PIPE)
	out = p.stdout.read()
	err = p.stderr.read()
	rc = p.wait()
	if rc != 0:
		print out
		print err
		if rc < 0:
			print "`%s' was killed with signal" % cmd, -rc
		else:
			print "`%s' exited with error code" % cmd, rc
		exit(5)
	if opts.time:
		m = re.search(r"^time for .*: (\d+)$", err)
		if not m:
			print "cmd:", cmd
			print "out:", out
			print "err:", err
			print "rc:", rc
			exit(5)
		time = int(m.group(1))
		return out, time
	else:
		return out, None

# get the result from the initial, reference file, without SIMD'izing anything
wsname = ws.name
ws.compile(outfile = "%s.database/Tmp/ref" % wsname,
		   outdir =  "%s.database/Tmp" % wsname,
		   CFLAGS = "-O0")
ref, ref_time = getout("./%s.database/Tmp/ref" % wsname)

module = ws[opts.function]
print "Module", module.name, "selected"
print "Initial code"
module.display()

# Magie ! The "128" is the size of registers
module.sac(128)

print "Simdized code"
module.display()

# Compile using the sequential (naïve) versions of SIMD instructions
ws.compile(outfile = "%s.database/Tmp/seq" % (wsname),
		   outdir =  "%s.database/Tmp" % (wsname))
seq, seq_time = getout("./%s.database/Tmp/seq" % wsname)

if seq != ref:
	print "seq ko"
	print "seq:", seq
	print "ref:", ref
	exit(3)
else:
	print "seq ok"

ws.sse_compile(outfile = "%s.database/Tmp/sse" % (wsname),
			   outdir =  "%s.database/Tmp" % (wsname))
sse, sse_time = getout("./%s.database/Tmp/sse" % wsname)

if sse != ref:
	print "sse ko"
	print "sse:", sse
	print "ref:", ref
	exit(3)
else:
	print "sse ok"

ws.close()

if opts.time:
    print "Run times: (ref, seq, sse):"
    print ref_time
    print seq_time
    print sse_time
