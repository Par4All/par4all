from pyps import workspace

with workspace('exception.c',name='exception') as ws:
	try: ws0=workspace(['exception.c'],name='exception')
	except RuntimeError: print "grrrrr, same workspace"

