from __future__ import with_statement # this is to work with python2.5

from pyps import workspace

with  workspace("pragma.c",name="pragma",deleteOnCreate=True) as w:
	w.props.POCC_COMPATIBILITY = True
	w.props.CONSTANT_PATH_EFFECTS = False
	w.all_functions.static_controlize()

