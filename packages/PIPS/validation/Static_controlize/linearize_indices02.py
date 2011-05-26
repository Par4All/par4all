from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace
import pocc

with  vworkspace() as w:
	w.props.POCC_COMPATIBILITY = True
	w.props.CONSTANT_PATH_EFFECTS = False
	w.all_functions.static_controlize()
	w.all_functions.pocc_prettyprinter()
	w.all_functions.display()
	w.all_functions.poccify() # pocc is not installed in the validation

