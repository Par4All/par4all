from __future__ import with_statement # this is to work with python2.5
#!/usr/bin/env python

# import everything so that a session looks like tpips one
from pyps import workspace

with workspace("properties2.f",deleteOnClose=True) as w:
	#Get foo function
	foo1 = w.fun.FOO1
	foo2 = w.fun.FOO2
	foo3 = w.fun.FOO3
	foo4 = w.fun.FOO4
	foo5 = w.fun.FOO5
	foo6 = w.fun.FOO6

	# the return type of this one should be void
	foo1.display (rc="c_printed_file")

	# the return type of this one should be plouch
	foo2.display (rc="c_printed_file",
		      DO_RETURN_TYPE_AS_TYPEDEF=True,
		      SET_RETURN_TYPE_AS_TYPEDEF_NEW_TYPE="plouch")

	# the return type of this one should be void if the context
	# has been restored
	foo3.display (rc="c_printed_file")

	# the return type of this one should be void
	foo4.display (rc="c_printed_file",
		      DO_RETURN_TYPE_AS_TYPEDEF=False)

	# the return type of this one should be void
	foo5.display (rc="c_printed_file")

	# the return type of this one should be the default value
	# i.e p4a_smth
	foo6.display (rc="c_printed_file",
		      DO_RETURN_TYPE_AS_TYPEDEF=True)
