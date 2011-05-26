#/usr/bin/env python
import sys
import re

usage= "usage: pipsmakerc-extract-loop-phase.py rc-file.tex properties.rc pipsdep.rc"

if len(sys.argv) < 4:
	print usage
	exit(1)

texfile = sys.argv[1]

#Read propdep file and convert it into a map.
input = open(sys.argv[3],"r")
lines = input.readlines()
input.close()

pipsdep= dict()
for line in lines:
	m = re.match(r"(.*?):\s*(.*)", line)
	
	p = m.group(1)
	
	deps = []
	if m.lastindex == 2:
		deps = re.split(" ", m.group(2))
		deps = deps[0:len(deps)-1]
	deps = map(lambda x: x.lower(), deps)
	pipsdep[p] = deps

#Read properties into a string
rcfile = sys.argv[2]
input = open(rcfile,"r")
lines = input.readlines()
input.close()

pipsprops = dict()
for line in lines:
	m = re.match("\s*(.*?)\s+(.*)", line)
	d = m.group(2)
	if d == "TRUE": d = "True"
	if d == "FALSE" : d = "False"
	pipsprops[m.group(1)] = d

#Read input tex file into a string
input = open(texfile,"r")
lines = input.readlines()
rc = "".join(lines)
input.close()


def printPythonMethod(name,doc):
	has_loop_label = False

	if name in pipsdep and len(pipsdep[name]) > 0:
		props = []
		for prop in pipsdep[name]:
			if prop == "loop_label":
				has_loop_label = True
				break
	if has_loop_label:
		print "\""+name+"\","


#Parse string documentation
doc_strings= re.findall(r'\\begin\{PipsPass\}(.*?)\\end\{PipsPass\}', rc, flags=re.M | re.S)

print "static char *loop_phases[] = {"
for dstr in doc_strings:
	m = re.match(r'\{([^\}]+)\}[\n]+(.*)', dstr, flags = re.M | re.S)
	printPythonMethod(m.group(1), m.group(2))
print "0\n};"
