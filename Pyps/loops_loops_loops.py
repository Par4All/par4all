from pyps import *
w = workspace(["loops_loops_loops.c"],verboseon=False)
m = w["looping"]

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
m.display()
