from pyps import *
workspace.delete("w0")
w0=workspace("basics0.c",name="w0",deleteOnClose=True)
w0.close()
try:
	w1=workspace("basics0.c",name="w0",deleteOnClose=True)
	w1.close()
except:
	print 'exception caught'
w2=workspace("basics0.c",deleteOnClose=True)
w2.close()

