import pyps
import workspace_gettime as gt
import subprocess

w = pyps.workspace(["hello.c"], parents=[gt.workspace])

exec_out = w.compile(CC="gcc", CFLAGS=" -O3 ",outdir="testdir",outfile="/tmp/a.out")

elapsed=[]
#print "running",exec_out
for i in range(0,20):
	t=-1
	while t<0:
		subprocess.call(exec_out)
		with open("_pyps_time.tmp", "r") as f:
			t = int(f.readline())
		#print "measured:" , t
	elapsed+=[t]
elapsed.sort()

print str(elapsed)
