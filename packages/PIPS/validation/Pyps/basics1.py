from __future__ import with_statement # this is to work with python2.5
#!/usr/bin/env python
from pyps import *
with workspace("basics1.c",deleteOnClose=True) as w:
	pass #do nothing, with statement wil automagically close everything
