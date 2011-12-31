from __future__ import with_statement # this is to work with python2.5
from pyps import workspace
with workspace("properties0.c",verbose=False,deleteOnClose=True) as w:
	print w.props.MUST_REGIONS
	w.props.MUST_REGIONS=True
	print w.props.must_regions
