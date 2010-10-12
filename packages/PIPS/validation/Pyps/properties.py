#!/usr/bin/env python

# import everything so that a session looks like tpips one
from pyps import workspace

with workspace("properties.c") as w:
	#Get foo function
	foo = w.fun.foo
	
	#Test for default value (should be True)
	print "Prop INLINING_PURGE_LABELS  is " + str(w.props.INLINING_PURGE_LABELS)
	
	#Should not purge labels
	foo.inlining(callers="bar",PURGE_LABELS=False)
	
	#Environment should have been restored, so PURGE_LABELS is True
	print "Prop INLINING_PURGE_LABELS  is " + str(w.props.INLINING_PURGE_LABELS)
	
	#Test if we success changing environment
	w.props.INLINING_PURGE_LABELS = False
	print "Prop INLINING_PURGE_LABELS  is " + str(w.props.INLINING_PURGE_LABELS)
	
	#Test keep it back to an other value
	w.props.INLINING_PURGE_LABELS = True
	print "Prop INLINING_PURGE_LABELS  is " + str(w.props.INLINING_PURGE_LABELS)
	
	#Should not purge labels
	foo.inlining(callers="foobar",PURGE_LABELS=False)
	
	#Environment should have been restored, so PURGE_LABELS is True
	print w.props.INLINING_PURGE_LABELS
	
	#Here, the call should purge labels
	foo.inlining(callers="malabar")
	
	w.all.display()
