import pyps
import re

""" add pragma filtering capability to pyps modules when imported """

pragma_re = re.compile(r'^ *#pragma +pyps +(.*)')
label_re = re.compile(r'^(\w)+:')

def p3(m):
	"""process `module' and apply #pragma as they go"""
	m.flag_loops()
	label=""
	runs=[]
	lines=m.code.split('\n')
	for line in lines:
		pm=pragma_re.match(line)
		lm=label_re.match(line)
		if pm:
			if label:runs.append('m.loops(label).'+pm.group(1))
			else:runs.append("m."+pm.group(1))
		elif lm:
			label=lm.group(1)
		else:
			label=""
	for r in runs:
		print "running",r
		eval(r)
	m.code=pragma_re.sub("",m.code)

pyps.module.p3=p3
pyps.modules.p3=lambda m:map(p3,m)
