import pyps
try :
	ws0=pyps.workspace(['exception.c'],name='exception')
	try :ws=pyps.workspace(['exception.c'],name='exception')
	except RuntimeError: print "grrrrr, same workspace"
	try :ws=pyps.workspace(['exception.c'],name='exception0')
	except RuntimeError: print "re grrrrr, not two workspaces at once"
	ws0.close()
	try :
		ws=pyps.workspace(['exception.c'],name='exception0')
		ws.close()
	except RuntimeError: print "should not happen!"
except RuntimeError: print "should not happen!"
