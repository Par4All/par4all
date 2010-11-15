from pyps import *
workspace.delete("w0")
w0=workspace("basics0.c",name="w0")
w0.close()
try:
	w1=workspace("basics0.c",name="w0")
	w1.close()
except:
	print 'exception caught'
w2=workspace("basics0.c")
w2.close()

