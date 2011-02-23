from __future__ import with_statement # this is to work with python2.5
from pyps import workspace
with workspace("loops_loops_loops.c",verbose=False,deleteOnClose=True) as w:
	m = w.fun.looping
	
	print "= first level loops"
	for l0 in m.loops():
		print l0.label
	
	print "= second level loops"
	for l0 in m.loops():
		print "== loops of" , l0.label
		for l1 in l0.loops():
			print l1.label
		
	print "= third level loops"
	for l0 in m.loops():
		for l1 in l0.loops():
			print "== loops of" , l1.label
			for l2 in l1.loops():
				print l2.label
	m.loops("l99995")
	m.display()
