from __future__ import with_statement # this is to work with python2.5
#!/usr/bin/env python

# import everything so that a session looks like tpips one
from pyps import workspace

with workspace("properties.c",deleteOnClose=True) as w:
	#Get foo function
	foo = w.fun.foo
	
	#Test for default value (should be True)
	print "Prop INLINING_USE_INITIALIZATION_LIST  is " + str(w.props.INLINING_USE_INITIALIZATION_LIST)
	
	#Should not purge labels
	foo.inlining(callers="bar",USE_INITIALIZATION_LIST=False)
	
	#Environment should have been restored, so USE_INITIALIZATION_LIST is True
	print "Prop INLINING_USE_INITIALIZATION_LIST  is " + str(w.props.INLINING_USE_INITIALIZATION_LIST)
	
	#Test if we success changing environment
	w.props.INLINING_USE_INITIALIZATION_LIST = False
	print "Prop INLINING_USE_INITIALIZATION_LIST  is " + str(w.props.INLINING_USE_INITIALIZATION_LIST)
	
	#Test keep it back to an other value
	w.props.INLINING_USE_INITIALIZATION_LIST = True
	print "Prop INLINING_USE_INITIALIZATION_LIST  is " + str(w.props.INLINING_USE_INITIALIZATION_LIST)
	
	#Should not purge labels
	foo.inlining(callers="foobar",USE_INITIALIZATION_LIST=False)
	
	#Environment should have been restored, so USE_INITIALIZATION_LIST is True
	print w.props.INLINING_USE_INITIALIZATION_LIST
	
	#Here, the call should purge labels
	foo.inlining(callers="malabar")
	
	w.all.display()
