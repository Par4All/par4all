#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import with_statement
import os,sys,re,subprocess

def parse_ops(code, op, f):
	matches = re.findall("(%%%%%s (\")?([a-zA-Z0-9 _./-]+)(\")?[:space:]*%%%%)" % op, code)
	retcode = code
	for m in matches:
		retcode = f(retcode, m[0], m[2])
	return retcode


def parse_include(code, org_code, arg):
	global _abspath
	hfile = os.path.join(_abspath,arg)
	print >>sys.stderr, "Includes %s..." % (hfile)
	with open(hfile, "r") as f:
		hcode = f.read()
	return code.replace(org_code, hcode)


def parse_gen_header(code, org_code, arg):
	global _abspath
	cfile = os.path.join(_abspath,arg)
	print >>sys.stderr, "Generate headers for %s using cproto..." % (cfile)
	p = subprocess.Popen(['cproto', cfile], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	(output, errors) = p.communicate()
	if p.returncode != 0:
		raise RuntimeError("cproto failed with return code %d. stderr says... :\n%s" % (p.returncode, errors))
	return code.replace(org_code, output)


if len(sys.argv) != 2:
	print >>sys.stderr, "Usage: %s path/to/_sac.py" % sys.argv[0]
	sys.exit(1)

_abspath = os.path.abspath(os.path.dirname(sys.argv[0])) # So that relative paths can be used !

_sac = sys.argv[1]
with open(_sac, "r") as f:
	saccode = f.read()

try:
	# Parse includes
	saccode = parse_ops(saccode, "include", parse_include)

	# Parse gen_header
	saccode = parse_ops(saccode, "gen_header", parse_gen_header)

except Exception,e:
	print >>sys.stderr, e
	sys.exit(1)

print saccode
sys.exit(0)
